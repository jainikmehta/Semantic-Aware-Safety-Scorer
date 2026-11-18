"""Corrected risk labels: the teacher's ratings, checked against the reviewed policy and the reviewer's own values.

The teacher's label files are never changed. The corrected set goes to <root>/labels/<teacher set>_corrected/ with the same
layout; every obstacle keeps the teacher's rating as `risk_teacher` and says what was done in `correction`:

    override    the reviewer gave this obstacle a value (review notes, `overrides.txt`)
    clipped     the rating was outside the band the policy gives this kind of obstacle and the teacher had recognised
                the obstacle: moved to the nearest edge of the band
    replaced    outside the band and the teacher described something else (it rated a neighbour that showed in the
                close-up, e.g. "person's legs" for a bucket behind a person): replaced by the median in-band rating of
                that kind of obstacle
    none        inside the band: the teacher's rating is kept

The bands (`corrections.bands` in configs/risk_config.yaml) are the reviewed policy. They decide which ratings are trusted,
never the value inside a band, which stays the teacher's.
"""

from __future__ import annotations

import json
import os
import re
import statistics

# words that show the teacher recognised the obstacle (synonyms included: a couch called "armchair" is still recognised
# as upholstered furniture, which is what its risk depends on)
RECOGNISED = {
    "bucket": "bucket barrel pail bin container can drum tub planter",
    "couch": "sofa couch settee loveseat armchair chair furniture upholstered",
    "armchair": "chair armchair seat sofa bench furniture upholstered",
    "ottoman": "ottoman stool footstool bench pouf seat chair sofa furniture upholstered",
    "bean bag": "bean beanbag pouf dome bag cushion ball inflatable foam fabric",
    "person": "person man worker human legs leg boot boots people woman",
    "gripper robot": "robot arm manipulator machine rover",
    "glass vase": "vase glass jar bottle urn transparent",
    "drinking glass": "glass cup tumbler mug jar bottle transparent",
}
NOTE = re.compile(r"^(\S+)#(\d+)\s+(.+?): teacher ([\d.]+), should be ([\d.]+)\s*$")


def base_kind(state: str) -> str:
    return state.split(" + ")[0].replace(" on the floor", "")


def recognised(state: str, name: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", name.lower())) & set(RECOGNISED[base_kind(state)].split()))


def band_for(state: str, bands: dict) -> tuple[float, float]:
    """The policy band of an obstacle state such as 'bucket + drinking glass'. Exact states first, then by kind."""
    if state in bands:
        return tuple(bands[state])
    kind = base_kind(state)
    if " + " in state:
        key = f"{kind} carrying fragile"
        return tuple(bands[key] if key in bands else bands["carrying fragile"])
    return tuple(bands[kind])


def read_overrides(path: str) -> dict[tuple[str, int], float]:
    """Reviewer values from the review page's notes: 'frame#mark  state: teacher x, should be y' per line."""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            m = NOTE.match(line.strip())
            if m:
                out[(m.group(1), int(m.group(2)))] = float(m.group(5))
    return out


def correct(records: list[dict], bands: dict, overrides: dict) -> tuple[list[dict], dict]:
    """Corrected copies of the label records and counts of what was done, per state."""
    inside: dict[str, list[float]] = {}
    for record in records:
        for mark in record["marks"]:
            risk = record["labels"][str(mark["mark"])]["risk"]
            low, high = band_for(mark["state"], bands)
            if low <= risk <= high and recognised(mark["state"], record["labels"][str(mark["mark"])]["name"]):
                inside.setdefault(mark["state"], []).append(risk)
    typical = {state: statistics.median(values) for state, values in inside.items()}
    counts: dict[str, dict[str, int]] = {}
    out = []
    for record in records:
        record = json.loads(json.dumps(record))
        for mark in record["marks"]:
            label = record["labels"][str(mark["mark"])]
            risk, state = label["risk"], mark["state"]
            low, high = band_for(state, bands)
            key = (record["frame"], mark["mark"])
            if key in overrides:
                final, how = overrides[key], "override"
            elif low <= risk <= high:
                final, how = risk, "none"
            elif recognised(state, label["name"]):
                final, how = min(max(risk, low), high), "clipped"
            else:
                final, how = typical.get(state, (low + high) / 2.0), "replaced"
            label["risk_teacher"], label["risk"], label["correction"] = risk, round(final, 3), how
            counts.setdefault(state, {}).setdefault(how, 0)
            counts[state][how] += 1
        out.append(record)
    missing = [k for k in overrides if not any(r["frame"] == k[0] for r in records)]
    if missing:
        raise ValueError(f"overrides for frames without labels: {missing}")
    return out, counts
