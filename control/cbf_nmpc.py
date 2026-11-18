"""Risk-aware CBF-NMPC for the TurtleBot 4: drives to the goal and keeps a distance from every obstacle that grows with its
semantic risk.

Model: unicycle with the velocities as states, so that the Create 3's acceleration limits are respected,

    x = [px, py, theta, v, w],  u = [a, alpha],   px' = v cos(theta), py' = v sin(theta), theta' = w, v' = a, w' = alpha

discretised with RK4 at dt over N steps. Every obstacle i is a rounded box: an oriented rectangle (centre c_i + k dt v_i,
yaw psi_i, half sides a_i, b_i) grown by a corner radius R_i, so a circle is a box of size zero and a sofa is one box
that does not bulge past its front. d_k is the robot's distance to the rectangle (smoothed at its corners and faces):

    contact   d_k >= rho + R_i + m0                                 hard: never touch, whatever the risk says
    risk CBF  h_{k+1} >= (1 - gamma(r_i)) h_k - gamma(r_i) s_{i,k},  s >= 0     h_k = d_k - (rho + R_i + m(r_i))

h is a distance in metres, so that a centimetre too close costs the same for a large and a small obstacle (with squared
distances the large one dominated, and the robot gave a small high-risk vase less room than a large bucket).

with the margin m(r) = m_low + (m_high - m_low) r and the decay rate gamma(r) = gamma_high - (gamma_high - gamma_low) r: a
riskier obstacle is kept further away, and the robot starts giving way earlier (a small gamma lets h shrink only slowly).
The slack s is penalised heavily, so the risk margin gives way only where the alley is too narrow for it, never the contact
constraint. s is scaled by gamma so that it measures how far the robot is inside the risk margin, in metres (a robot
holding a constant h < 0 needs s = -h), whatever gamma is; its cost grows with the risk, (1/2 + r_i)(w s^2 + w' s). With
the slack unscaled, a robot squeezed between two obstacles paid gamma^2 h^2 per obstacle, and since a risky obstacle has
the smaller gamma, intruding into its margin was the cheaper one: in a gate of a bucket with a glass and an empty bucket,
it passed both at the same distance. Walls are bounds on py. Solved with IPOPT, warm-started from the previous solution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass
class Obstacle:
    """A rounded box on the floor: the rectangle (centre x, y, yaw, half sides hx, hy) grown by radius. A circle has
    hx = hy = 0."""
    x: float
    y: float
    radius: float
    risk: float
    vx: float = 0.0
    vy: float = 0.0
    name: str = ""
    yaw: float = 0.0
    hx: float = 0.0
    hy: float = 0.0

    def distance(self, px: float, py: float) -> float:
        """Distance from a point to the obstacle's boundary (negative inside)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx, ly = c * (px - self.x) + s * (py - self.y), -s * (px - self.x) + c * (py - self.y)
        qx, qy = abs(lx) - self.hx, abs(ly) - self.hy
        return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - self.radius

    def outline(self, extra: float = 0.0, arc: int = 6) -> list[tuple[float, float]]:
        """Polygon of the boundary grown by extra (for drawing and rasterising)."""
        r = self.radius + extra
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = []
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            start = math.pi / 2 * k
            for j in range(arc + 1):
                a = start + math.pi / 2 * j / arc
                u, v = sx * self.hx + r * math.cos(a), sy * self.hy + r * math.sin(a)
                pts.append((self.x + c * u - s * v, self.y + s * u + c * v))
        return pts


class CbfNmpc:
    def __init__(self, cfg: dict, robot_radius: float, limits: dict, y_bounds: tuple[float, float]):
        c = cfg
        self.cfg = c
        self.N, self.dt = int(c["horizon"]), float(c["dt"])
        self.M = int(c["max_obstacles"])
        self.rho = robot_radius
        self.limits = limits
        n, m, M, N = 5, 2, self.M, self.N
        opti = ca.Opti()
        X, U, S = opti.variable(n, N + 1), opti.variable(m, N), opti.variable(M, N)
        x0, goal = opti.parameter(n), opti.parameter(2)
        # obs rows: x, y, vx, vy, cos yaw, sin yaw, half sides a, b, contact radius (rho + R + m0), risk radius
        # (rho + R + m(r)), gamma(r), slack weight

        def smax(a, b):  # smooth max: corners and faces of the rectangle without kinks for IPOPT
            return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + 1e-5))

        def box_distance(px, py, i, ox, oy):
            dx, dy = px - ox, py - oy
            lx, ly = obs[4, i] * dx + obs[5, i] * dy, -obs[5, i] * dx + obs[4, i] * dy
            qx, qy = ca.sqrt(lx ** 2 + 1e-6) - obs[6, i], ca.sqrt(ly ** 2 + 1e-6) - obs[7, i]
            ox_, oy_ = smax(qx, 0), smax(qy, 0)
            return ca.sqrt(ox_ ** 2 + oy_ ** 2 + 1e-6) - smax(-smax(qx, qy), 0)

        def f(x, u):
            return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), x[4], u[0], u[1])

        cost = 0
        w = c["weights"]
        for k in range(N):
            k1 = f(X[:, k], U[:, k])
            k2 = f(X[:, k] + self.dt / 2 * k1, U[:, k])
            k3 = f(X[:, k] + self.dt / 2 * k2, U[:, k])
            k4 = f(X[:, k] + self.dt * k3, U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
            e = X[0:2, k + 1] - goal
            cost += w["position"] * ca.sumsqr(e) + w["lateral"] * X[1, k + 1] ** 2 + w["accel"] * U[0, k] ** 2 \
                + w["alpha"] * U[1, k] ** 2 + w["omega"] * X[4, k + 1] ** 2
        cost += w["terminal"] * ca.sumsqr(X[0:2, N] - goal)
        obs = opti.parameter(12, M)  # see below; row 11: the slack weight (1/2 + r)
        for i in range(M):
            cost += obs[11, i] * (w["slack"] * ca.sumsqr(S[i, :]) + w["slack_linear"] * ca.sum2(S[i, :]))
        opti.minimize(cost)

        opti.subject_to(X[:, 0] == x0)
        # from the second state on: the first is the measured one, which can be a little over the limits (wheel slip,
        # the speed cap below the robot's own), and bounding it made the problem infeasible in 40 % of the solves
        opti.subject_to(opti.bounded(-c["max_reverse"], X[3, 1:], limits["max_linear_vel"]))
        opti.subject_to(opti.bounded(-limits["max_angular_vel"], X[4, 1:], limits["max_angular_vel"]))
        opti.subject_to(opti.bounded(-limits["max_linear_accel"], U[0, :], limits["max_linear_accel"]))
        opti.subject_to(opti.bounded(-limits["max_angular_accel"], U[1, :], limits["max_angular_accel"]))
        opti.subject_to(opti.bounded(y_bounds[0] + robot_radius + c["wall_margin"], X[1, 1:], y_bounds[1] - robot_radius - c["wall_margin"]))
        opti.subject_to(ca.vec(S) >= 0)
        for i in range(M):
            for k in range(N):
                ox_k, oy_k = obs[0, i] + k * self.dt * obs[2, i], obs[1, i] + k * self.dt * obs[3, i]
                ox_n, oy_n = ox_k + self.dt * obs[2, i], oy_k + self.dt * obs[3, i]
                d_k = box_distance(X[0, k], X[1, k], i, ox_k, oy_k)
                d_n = box_distance(X[0, k + 1], X[1, k + 1], i, ox_n, oy_n)
                opti.subject_to(d_n >= obs[8, i])
                h_k, h_n = d_k - obs[9, i], d_n - obs[9, i]
                opti.subject_to(h_n >= (1 - obs[10, i]) * h_k - obs[10, i] * S[i, k])
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": int(c["max_iter"]), "tol": 1e-4, "acceptable_tol": 1e-2,
                     "acceptable_iter": 3, "warm_start_init_point": "yes", "mu_strategy": "adaptive", "sb": "yes"})
        self.opti, self.X, self.U, self.S, self.x0, self.goal, self.obs = opti, X, U, S, x0, goal, obs
        self._guess: tuple[np.ndarray, np.ndarray] | None = None
        self.last = {}

    # ------------------------------------------------------------------------------------------------- risk mapping
    def margin(self, risk: float) -> float:
        c = self.cfg["risk"]
        return c["margin_low"] + (c["margin_high"] - c["margin_low"]) * float(np.clip(risk, 0.0, 1.0))

    def gamma(self, risk: float) -> float:
        c = self.cfg["risk"]
        return c["gamma_high"] - (c["gamma_high"] - c["gamma_low"]) * float(np.clip(risk, 0.0, 1.0))

    # ------------------------------------------------------------------------------------------------- solve
    def solve(self, state: np.ndarray, goal: tuple[float, float], obstacles: list[Obstacle]) -> tuple[float, float]:
        """state [x, y, theta, v, w]; returns the velocity command (v, w) for the next step."""
        start = time.perf_counter()
        x, y = float(state[0]), float(state[1])
        near = sorted(obstacles, key=lambda o: o.distance(x, y))[:self.M]
        params = np.zeros((12, self.M))
        params[0, :], params[1, :] = x + 100.0, y + 100.0  # unused slots: far away
        params[4, :], params[8, :], params[9, :], params[10, :], params[11, :] = 1.0, 0.1, 0.1, 1.0, 1.0
        m0 = self.cfg["risk"]["contact_margin"]
        for i, o in enumerate(near):
            params[:, i] = [o.x, o.y, o.vx, o.vy, math.cos(o.yaw), math.sin(o.yaw), o.hx, o.hy, self.rho + o.radius + m0,
                            self.rho + o.radius + self.margin(o.risk), self.gamma(o.risk), 0.5 + float(np.clip(o.risk, 0.0, 1.0))]
        self.opti.set_value(self.x0, np.asarray(state, float))
        self.opti.set_value(self.goal, np.asarray(goal, float))
        self.opti.set_value(self.obs, params)
        if self._guess is not None:
            Xg, Ug = self._guess
            self.opti.set_initial(self.X, np.hstack([Xg[:, 1:], Xg[:, -1:]]))
            self.opti.set_initial(self.U, np.hstack([Ug[:, 1:], Ug[:, -1:]]))
        else:
            self.opti.set_initial(self.X, np.tile(np.asarray(state, float)[:, None], (1, self.N + 1)))
        status = "ok"
        try:
            sol = self.opti.solve()
            Xs, Us, Ss = sol.value(self.X), sol.value(self.U), sol.value(self.S)
        except RuntimeError:  # not converged: use the last iterate if it is usable, else brake
            status = "not converged"
            Xs, Us, Ss = self.opti.debug.value(self.X), self.opti.debug.value(self.U), self.opti.debug.value(self.S)
            if not np.all(np.isfinite(Xs)):
                self._guess = None
                self.last = {"status": "failed", "ms": (time.perf_counter() - start) * 1e3, "plan": None}
                return 0.0, 0.0
        self._guess = (np.asarray(Xs), np.asarray(Us))
        Xs = np.asarray(Xs)
        m0 = self.cfg["risk"]["contact_margin"]
        if status != "ok" and any(o.distance(px, py) < self.rho + m0 for o in near for px, py in Xs[:2, 1:6].T):
            status = "braking"  # not converged, and the last iterate touches something: stop instead of following it
            v, w = 0.0, 0.0
        else:
            v, w = float(Xs[3, 1]), float(Xs[4, 1])  # the planned velocities one step ahead
        v, filtered = self.emergency_filter(np.asarray(state, float), v, w, near)
        self.last = {"status": status, "ms": (time.perf_counter() - start) * 1e3, "plan": Xs[:2].T.copy(),
                     "slack": float(np.max(Ss)) if np.size(Ss) else 0.0, "obstacles": near, "emergency": filtered}
        return v, w

    def emergency_filter(self, state: np.ndarray, v: float, w: float, obstacles: list[Obstacle], horizon: float = 0.5) -> tuple[float, bool]:
        """Last line of defence, outside the optimiser: roll the command out for horizon seconds (the robot keeps its
        velocity and turn rate); if that brings the robot inside the contact distance of an obstacle and closer to it than
        it is now, drop the forward speed to zero (turning stays allowed). Driving along an obstacle at the contact
        distance is allowed: blocking that froze the robot in a tight gate."""
        if v <= 0.0:
            return v, False
        m0 = self.cfg["risk"]["contact_margin"]
        x, y, th = state[0], state[1], state[2]
        now = [o.distance(x, y) for o in obstacles]
        dt = 0.05
        for k in range(1, int(horizon / dt) + 1):
            x, y, th = x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt
            for o, d0 in zip(obstacles, now):
                d = o.distance(x - o.vx * k * dt, y - o.vy * k * dt)  # obstacles move on
                if d < self.rho + m0 and d < d0 - 1e-3:
                    return 0.0, True
        return v, False
