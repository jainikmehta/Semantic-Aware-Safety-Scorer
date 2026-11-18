"""Summarise the teacher's labels and build the review sheet.

    python scripts/build_risk_review.py

Prints, per kind of obstacle and arrangement, how the teacher rated it against the band the rubric implies
(`review.expected` in configs/risk_config.yaml) and how often its attributes match what the simulator knows. Writes
<root>/review/v<rubric version>/review.html (the review frames with their labels, self-contained), summary.json and
labels.csv (one row per rated obstacle).
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import html
import json
import os
import statistics
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from distillation import teacher as T  # noqa: E402

SOFT = ("couch", "armchair", "ottoman", "bean bag")
TOLERANCE = 0.05  # a rating this far outside its band still counts as inside


def expected_band(state: str, expected: dict) -> tuple[float, float]:
    base = state.split(" + ")[0]
    if " + " in state:
        return tuple(expected["person" if base == "person" else "carrying_fragile"])
    return tuple(expected[base])


def rows_from(records: list[dict], expected: dict) -> list[dict]:
    rows = []
    for record in records:
        for mark in record["marks"]:
            label = record["labels"][str(mark["mark"])]
            low, high = expected_band(mark["state"], expected)
            base = mark["state"].split(" + ")[0]
            rows.append({
                "frame": record["frame"], "split": record["split"], "scenario": record["scenario"], "mark": mark["mark"],
                "object": mark["name"], "state": mark["state"], "carried_hidden": ", ".join(mark["carried_hidden"]),
                "pixels": mark["pixels"], "distance": mark["distance"], "risk": label["risk"], "band_low": low, "band_high": high,
                "inside": low - TOLERANCE <= label["risk"] <= high + TOLERANCE, "name": label["name"], "carries": label["carries"],
                "soft": label["soft"], "fragile": label["fragile"], "carries_fragile": label["carries_fragile"], "living": label["living"],
                "reason": label["reason"],
                "truth_soft": base in SOFT, "truth_carries_fragile": bool(mark["carried"]), "truth_living": base == "person",
                "truth_fragile": base in ("glass vase on the floor", "drinking glass on the floor"),
            })
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for state in sorted({r["state"] for r in rows}, key=lambda s: (statistics.mean(r["risk"] for r in rows if r["state"] == s), s)):
        group = [r for r in rows if r["state"] == state]
        risks = sorted(r["risk"] for r in group)
        out.append({"state": state, "count": len(group), "mean": statistics.mean(risks), "median": statistics.median(risks),
                    "std": statistics.pstdev(risks), "min": risks[0], "max": risks[-1], "band": [group[0]["band_low"], group[0]["band_high"]],
                    "inside": sum(r["inside"] for r in group) / len(group),
                    "sees_carried": sum(r["carries_fragile"] == r["truth_carries_fragile"] for r in group) / len(group)})
    return out


def attribute_accuracy(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for name in ("soft", "fragile", "carries_fragile", "living"):
        positives = [r for r in rows if r[f"truth_{name}"]]
        negatives = [r for r in rows if not r[f"truth_{name}"]]
        out[name] = {"recall": sum(r[name] for r in positives) / max(len(positives), 1), "positives": len(positives),
                     "false_alarms": sum(r[name] for r in negatives) / max(len(negatives), 1), "negatives": len(negatives)}
    return out


def print_summary(summary: list[dict], attributes: dict, rows: list[dict]) -> None:
    print(f"\n{len(rows)} rated obstacles in {len({r['frame'] for r in rows})} frames\n")
    print(f"{'obstacle':<38}{'n':>6}{'mean':>7}{'std':>6}{'min':>6}{'max':>6}   {'rubric band':<12}{'inside':>8}{'carried seen right':>20}")
    for s in summary:
        print(f"{s['state']:<38}{s['count']:>6}{s['mean']:>7.2f}{s['std']:>6.2f}{s['min']:>6.2f}{s['max']:>6.2f}   "
              f"{s['band'][0]:.2f}-{s['band'][1]:.2f}   {s['inside']:>7.0%}{s['sees_carried']:>20.0%}")
    print(f"\n{'attribute':<18}{'found when true':>18}{'claimed when false':>22}")
    for name, a in attributes.items():
        print(f"{name:<18}{a['recall']:>11.0%} of {a['positives']:<4}{a['false_alarms']:>15.0%} of {a['negatives']:<5}")


# ----------------------------------------------------------------------------------------------------- page
def risk_class(risk: float) -> str:
    return "r0" if risk < 0.15 else "r1" if risk < 0.35 else "r2" if risk < 0.6 else "r3" if risk < 0.85 else "r4"


def page(cfg: dict, root: str, review: list[dict], summary: list[dict], attributes: dict, rows: list[dict], width: int) -> str:
    e = html.escape
    total_frames = len({r["frame"] for r in rows})
    cards = []
    for record in review:
        rgb, ids = T.load_images(root, record["frame"])
        image = T.draw_marks(rgb, ids, record["marks"], 2, record["labels"])
        image = cv2.resize(image, (width, width * 3 // 4), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 76])
        source = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()
        frame_rows = [r for r in rows if r["frame"] == record["frame"]]
        lines = []
        for r in frame_rows:
            color = "#%02x%02x%02x" % T.MARK_COLORS[(r["mark"] - 1) % len(T.MARK_COLORS)]
            tags = "".join(f'<span class="tag">{t}</span>' for t, on in (("soft", r["soft"]), ("fragile", r["fragile"]),
                                                                           ("carries fragile", r["carries_fragile"]), ("living", r["living"])) if on)
            hidden = f' <span class="muted">({e(r["carried_hidden"])} on it, hidden in this view)</span>' if r["carried_hidden"] else ""
            key = f'{r["frame"]}#{r["mark"]}'
            lines.append(
                f'<tr data-state="{e(r["state"])}" data-inside="{int(r["inside"])}">'
                f'<td><span class="num" style="--mark:{color}">{r["mark"]}</span></td>'
                f'<td><div class="truth">{e(r["state"])}{hidden}</div>'
                f'<div class="said">Teacher: {e(r["name"])}; carries {e(r["carries"])}</div>'
                f'<div class="reason">{e(r["reason"])}</div><div class="tags">{tags}</div></td>'
                f'<td class="score"><span class="risk {risk_class(r["risk"])}">{r["risk"]:.2f}</span>'
                f'<div class="band {"" if r["inside"] else "off"}">band {r["band_low"]:.2f}&ndash;{r["band_high"]:.2f}</div></td>'
                f'<td class="fix"><label class="flag"><input type="checkbox" id="flag-{e(key)}" data-key="{e(key)}" data-state="{e(r["state"])}" '
                f'data-risk="{r["risk"]:.2f}"> wrong</label>'
                f'<input type="number" min="0" max="1" step="0.05" placeholder="should be" id="fix-{e(key)}" data-fix="{e(key)}" aria-label="Corrected risk"></td></tr>')
        states = " | ".join(sorted({r["state"] for r in frame_rows}))
        cards.append(
            f'<article class="card" data-states="{e(states)}" data-off="{int(any(not r["inside"] for r in frame_rows))}">'
            f'<img loading="lazy" src="{source}" alt="Camera frame {e(record["frame"])} with numbered obstacles" width="{width}" height="{width * 3 // 4}">'
            f'<div class="body"><div class="meta"><span class="frame">{e(record["frame"])}</span><span>{e(record["split"].replace("_", " "))}</span>'
            f'<span>{e(record["scenario"].replace("_", " "))} scene</span></div>'
            f'<div class="tablewrap"><table class="labels"><tbody>{"".join(lines)}</tbody></table></div></div></article>')

    summary_rows = "".join(
        f'<tr><td>{e(s["state"])}</td><td class="n">{s["count"]}</td>'
        f'<td class="n"><span class="risk {risk_class(s["mean"])}">{s["mean"]:.2f}</span></td><td class="n">{s["std"]:.2f}</td>'
        f'<td class="n">{s["min"]:.2f}&ndash;{s["max"]:.2f}</td><td class="n">{s["band"][0]:.2f}&ndash;{s["band"][1]:.2f}</td>'
        f'<td class="n {"warn" if s["inside"] < 0.8 else ""}">{s["inside"]:.0%}</td>'
        f'<td class="range"><svg viewBox="0 0 200 18" role="img" aria-label="Ratings from {s["min"]:.2f} to {s["max"]:.2f}, mean {s["mean"]:.2f}">'
        f'<rect class="axis" x="0" y="8" width="200" height="2"></rect>'
        f'<rect class="bandbar" x="{200 * s["band"][0]:.1f}" y="3" width="{200 * (s["band"][1] - s["band"][0]):.1f}" height="12"></rect>'
        f'<rect class="spread" x="{200 * s["min"]:.1f}" y="7" width="{max(200 * (s["max"] - s["min"]), 2):.1f}" height="4"></rect>'
        f'<circle class="dot" cx="{200 * s["mean"]:.1f}" cy="9" r="4"></circle></svg></td></tr>' for s in summary)
    attribute_rows = "".join(
        f'<tr><td>{e(name.replace("_", " "))}</td><td class="n">{a["recall"]:.0%} <span class="muted">of {a["positives"]}</span></td>'
        f'<td class="n">{a["false_alarms"]:.0%} <span class="muted">of {a["negatives"]}</span></td></tr>' for name, a in attributes.items())
    options = "".join(f'<option value="{e(s["state"])}">{e(s["state"])} ({s["count"]})</option>' for s in sorted(summary, key=lambda s: s["state"]))
    rubric = e(cfg["rubric"]["system"].strip())
    inside_all = sum(r["inside"] for r in rows) / max(len(rows), 1)
    return TEMPLATE.format(
        version=cfg["rubric"]["version"], model=e(cfg["teacher"]["model"]), obstacles=len(rows), frames=total_frames, shown=len(review),
        inside=f"{inside_all:.0%}", summary_rows=summary_rows, attribute_rows=attribute_rows, options=options, rubric=rubric,
        cards="\n".join(cards), hints="on" if cfg["teacher"].get("hints") else "off")


TEMPLATE = """<title>Risk Label Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column of summary tables, then a two-column grid of frame cards (image above its label table). */
:root {{
  --bg: #f3f5f7; --surface: #ffffff; --fg: #1b232c; --muted: #5f6b78; --line: #d5dbe2; --accent: #1f5fa8;
  --r0: #2f7d4f; --r1: #6f8a2a; --r2: #b07a12; --r3: #c4551c; --r4: #b3262e; --chip: #e8ecf1; --warn: #b3262e;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12171d; --surface: #1a212a; --fg: #e4e9ee; --muted: #98a4b1; --line: #2d3742; --accent: #7bb3ee;
  --r0: #5fc487; --r1: #a9c25a; --r2: #e3ad45; --r3: #f08a4e; --r4: #f0666c; --chip: #252e39; --warn: #f0666c; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12171d; --surface: #1a212a; --fg: #e4e9ee; --muted: #98a4b1; --line: #2d3742; --accent: #7bb3ee;
  --r0: #5fc487; --r1: #a9c25a; --r2: #e3ad45; --r3: #f08a4e; --r4: #f0666c; --chip: #252e39; --warn: #f0666c; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px; line-height: 1.5;
  padding-block: 28px 96px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 1240px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }}
h1 {{ font-size: 26px; font-weight: 600; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 17px; font-weight: 600; margin: 0 0 10px; }}
p {{ margin: 0; max-width: 68ch; }}
.lede {{ color: var(--muted); margin-top: 6px; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 14px; font-family: var(--mono); font-size: 13px; color: var(--muted); }}
.facts b {{ color: var(--fg); font-weight: 500; }}
.muted {{ color: var(--muted); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
.stats th, .stats td {{ text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--line); vertical-align: middle; white-space: nowrap; }}
.stats th {{ font-size: 12px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
.stats .n {{ font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; }}
.stats .warn {{ color: var(--warn); font-weight: 500; }}
.range svg {{ width: 200px; height: 18px; display: block; }}
.axis {{ fill: var(--line); }} .bandbar {{ fill: var(--accent); opacity: 0.22; }} .spread {{ fill: var(--muted); }} .dot {{ fill: var(--fg); }}
.legend {{ font-size: 13px; color: var(--muted); margin-top: 8px; }}
.risk {{ font-family: var(--mono); font-weight: 500; font-variant-numeric: tabular-nums; }}
.r0 {{ color: var(--r0); }} .r1 {{ color: var(--r1); }} .r2 {{ color: var(--r2); }} .r3 {{ color: var(--r3); }} .r4 {{ color: var(--r4); }}
details {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 10px 0; }}
summary {{ cursor: pointer; font-weight: 500; }}
pre {{ white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px; line-height: 1.55; margin: 12px 0 4px; max-width: 92ch; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; }}
select, input[type="number"] {{ font: inherit; font-size: 14px; color: var(--fg); background: var(--surface); border: 1px solid var(--line); border-radius: 4px; padding: 5px 8px; }}
select {{ max-width: 100%; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 20px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; min-width: 0; }}
.card img {{ display: block; width: 100%; height: auto; }}
.body {{ padding: 12px 14px 14px; display: flex; flex-direction: column; gap: 8px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--mono); font-size: 12px; color: var(--muted); }}
.meta .frame {{ color: var(--fg); }}
.tablewrap {{ overflow-x: auto; }}
.labels td {{ padding: 9px 10px 9px 0; border-top: 1px solid var(--line); vertical-align: top; }}
.labels tr:first-child td {{ border-top: 0; }}
.num {{ display: inline-flex; align-items: center; justify-content: center; min-width: 22px; height: 22px; border: 2px solid var(--mark); border-radius: 3px;
  font-family: var(--mono); font-size: 13px; font-weight: 500; }}
.truth {{ font-weight: 500; }}
.said, .reason {{ font-size: 13.5px; color: var(--muted); }}
.reason {{ color: var(--fg); }}
.tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }}
.tag {{ font-size: 11.5px; background: var(--chip); border-radius: 3px; padding: 1px 6px; }}
.score {{ white-space: nowrap; text-align: right; }}
.score .risk {{ font-size: 19px; }}
.band {{ font-family: var(--mono); font-size: 11.5px; color: var(--muted); }}
.band.off {{ color: var(--warn); }}
.fix {{ white-space: nowrap; }}
.fix input[type="number"] {{ width: 92px; display: block; margin-top: 4px; font-size: 13px; }}
.flag {{ font-size: 13px; display: inline-flex; gap: 5px; align-items: center; cursor: pointer; }}
.bar {{ position: fixed; left: 0; right: 0; bottom: 0; background: var(--surface); border-top: 1px solid var(--line);
  padding: 10px max(16px, env(safe-area-inset-right, 0px)) calc(10px + env(safe-area-inset-bottom, 0px)) max(16px, env(safe-area-inset-left, 0px)); }}
.bar > div {{ max-width: 1240px; margin: 0 auto; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; }}
button {{ font: inherit; font-weight: 500; color: var(--surface); background: var(--accent); border: 0; border-radius: 4px; padding: 7px 14px; cursor: pointer; }}
button.quiet {{ background: transparent; color: var(--accent); border: 1px solid var(--line); }}
#notes {{ width: 100%; min-height: 110px; font-family: var(--mono); font-size: 12.5px; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 8px; }}
</style>
<main>
  <header>
    <h1>Risk labels from the teacher model</h1>
    <p class="lede">Each camera frame went to {model} with the rubric below. It rated every numbered obstacle from 0 (harmless to bump into) to 1 (worst). Check whether the numbers match your judgement before the student model is trained on them.</p>
    <div class="facts"><span><b>{obstacles}</b> rated obstacles</span><span><b>{frames}</b> labelled frames</span><span><b>{shown}</b> frames shown here</span>
      <span><b>{inside}</b> inside the rubric's band</span><span>rubric v<b>{version}</b></span><span>scene hints <b>{hints}</b></span></div>
  </header>

  <section>
    <h2>Ratings by obstacle and what it carries</h2>
    <div class="scroll"><table class="stats">
      <thead><tr><th>Obstacle (from the simulator)</th><th>Count</th><th>Mean risk</th><th>Std</th><th>Range</th><th>Rubric band</th><th>Inside band</th><th>0 to 1</th></tr></thead>
      <tbody>{summary_rows}</tbody></table></div>
    <p class="legend">Shaded: the band the rubric implies for that obstacle. Bar: lowest to highest rating. Dot: mean. "Inside band" allows 0.05 either side. The statistics cover every labelled frame, not only the ones shown.</p>
  </section>

  <section>
    <h2>Does the teacher see what the simulator knows?</h2>
    <div class="scroll"><table class="stats" style="width:auto">
      <thead><tr><th>Attribute</th><th>Said true when true</th><th>Said true when false</th></tr></thead>
      <tbody>{attribute_rows}</tbody></table></div>
    <p class="legend">"Carries fragile" is true in the simulator when a glass or vase stands on the obstacle, or is held by it, and is visible in that frame.</p>
  </section>

  <details><summary>The rubric the teacher was given</summary><pre>{rubric}</pre></details>

  <section>
    <h2>Frames</h2>
    <div class="controls">
      <label>Obstacle <select id="state"><option value="">all</option>{options}</select></label>
      <label class="flag"><input type="checkbox" id="offonly"> only frames with a rating outside its band</label>
      <span class="muted" id="count"></span>
    </div>
  </section>
  <div class="grid" id="grid">
{cards}
  </div>
  <section id="notesbox" hidden>
    <h2>Your review notes</h2>
    <p class="muted" style="margin-bottom:8px">Paste these into the chat with Claude.</p>
    <textarea id="notes" readonly aria-label="Review notes"></textarea>
  </section>
</main>
<div class="bar"><div><span id="status">Tick "wrong" on any rating you disagree with, and enter what it should be if you like.</span>
  <span style="display:flex;gap:8px"><button class="quiet" id="clear" type="button">Clear marks</button><button id="copy" type="button">Copy review notes</button></span></div></div>
<script>
(function () {{
  var KEY = "risk-review-v{version}";
  var saved = {{}};
  try {{ saved = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ saved = {{}}; }}
  var flags = Array.prototype.slice.call(document.querySelectorAll("input[data-key]"));
  var fixes = Array.prototype.slice.call(document.querySelectorAll("input[data-fix]"));
  flags.forEach(function (f) {{ f.checked = !!(saved[f.dataset.key] && saved[f.dataset.key].wrong); }});
  fixes.forEach(function (f) {{ var s = saved[f.dataset.fix]; if (s && s.fix !== undefined) f.value = s.fix; }});
  function store() {{
    saved = {{}};
    flags.forEach(function (f) {{ if (f.checked) saved[f.dataset.key] = {{ wrong: true }}; }});
    fixes.forEach(function (f) {{ if (f.value !== "") {{ saved[f.dataset.fix] = saved[f.dataset.fix] || {{ wrong: true }}; saved[f.dataset.fix].fix = f.value; }} }});
    try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}}
    var n = Object.keys(saved).length;
    document.getElementById("status").textContent = n ? n + " rating" + (n > 1 ? "s" : "") + " marked wrong." : 'Tick "wrong" on any rating you disagree with, and enter what it should be if you like.';
  }}
  function notes() {{
    var lines = [];
    flags.forEach(function (f) {{
      var s = saved[f.dataset.key];
      if (s) lines.push(f.dataset.key + "  " + f.dataset.state + ": teacher " + f.dataset.risk + (s.fix !== undefined ? ", should be " + s.fix : ", wrong"));
    }});
    return lines.length ? "Risk label review, rubric v{version}:\\n" + lines.join("\\n") : "Risk label review, rubric v{version}: no ratings marked wrong.";
  }}
  flags.concat(fixes).forEach(function (el) {{ el.addEventListener("input", function () {{
    if (el.dataset.fix && el.value !== "") {{ var box = document.getElementById("flag-" + el.dataset.fix); if (box) box.checked = true; }}
    store();
  }}); }});
  document.getElementById("copy").addEventListener("click", function () {{
    var text = notes(), box = document.getElementById("notesbox"), area = document.getElementById("notes");
    area.value = text; box.hidden = false;
    var done = function () {{ document.getElementById("status").textContent = "Copied. Paste the notes into the chat with Claude."; }};
    var fallback = function () {{ area.focus(); area.select(); document.getElementById("status").textContent = "Select and copy the notes at the bottom of the page."; box.scrollIntoView(); }};
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
  }});
  document.getElementById("clear").addEventListener("click", function () {{
    flags.forEach(function (f) {{ f.checked = false; }}); fixes.forEach(function (f) {{ f.value = ""; }}); store();
  }});
  var state = document.getElementById("state"), off = document.getElementById("offonly");
  function filter() {{
    var shown = 0;
    Array.prototype.forEach.call(document.querySelectorAll(".card"), function (card) {{
      var ok = (!state.value || card.dataset.states.split(" | ").indexOf(state.value) >= 0) && (!off.checked || card.dataset.off === "1");
      card.hidden = !ok; if (ok) shown += 1;
    }});
    document.getElementById("count").textContent = shown + " frames";
  }}
  state.addEventListener("change", filter); off.addEventListener("change", filter);
  store(); filter();
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise the teacher's labels and build the review sheet.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--image-width", type=int, default=800)
    parser.add_argument("--labels", default=None, help="Label set under <root>/labels/ (default: the current rubric's), e.g. v1_hints_corrected.")
    args = parser.parse_args()
    cfg = T.load_config(args.config)
    root = T.dataset_root(cfg, args.root)
    label_dir = os.path.join(root, "labels", args.labels) if args.labels else T.Teacher(cfg, root).label_dir
    records = []
    for path in sorted(glob.glob(os.path.join(label_dir, "*", "*", "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    if not records:
        sys.exit(f"no labels in {label_dir}: run scripts/label_risk_dataset.py first")
    rows = rows_from(records, cfg["review"]["expected"])
    summary, attributes = summarise(rows), attribute_accuracy(rows)
    print_summary(summary, attributes, rows)

    out = os.path.join(root, "review", os.path.basename(label_dir))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"obstacles": len(rows), "frames": len(records), "by_state": summary, "attributes": attributes}, f, indent=1)
    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_frame = {r["frame"]: r for r in records}
    review_path = os.path.join(root, "review_frames.json")
    chosen = json.load(open(review_path)) if os.path.exists(review_path) else []
    review = [by_frame[f] for f in chosen if f in by_frame] or records[:cfg["review"]["frames"]]
    with open(os.path.join(out, "review.html"), "w") as f:
        f.write(page(cfg, root, review, summary, attributes, rows, args.image_width))
    print(f"\nReview sheet: {os.path.join(out, 'review.html')} ({len(review)} frames)\nPer-obstacle table: {os.path.join(out, 'labels.csv')}")


if __name__ == "__main__":
    main()
