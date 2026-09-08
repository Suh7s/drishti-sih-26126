"""Grid route planning and conservative command supervision, in metres and seconds.

The planner never receives a simulator world. It consumes an estimated GridMap.
Circular footprint is conservative for a small rover, but must be sized to Spot's
swept body/leg envelope before real physics validation. No formal safety guarantee.
"""
from dataclasses import dataclass
import heapq
import math
import numpy as np


@dataclass
class Config:
    resolution: float = 0.25
    radius: float = 0.55
    margin: float = 0.15
    max_speed: float = 0.45
    max_yaw_rate: float = 0.65
    max_accel: float = 0.45
    braking_accel: float = 0.6
    sensor_timeout: float = 0.6
    min_quality: float = 0.30
    cautious_quality: float = 0.65
    goal_tolerance: float = 0.5
    risk_weight: float = 5.0
    unknown_weight: float = 8.0
    support_max_age: float = 4.0
    slope_weight: float = 3.0
    max_slope_rad: float = 0.55
    obstacle_decay_age: float = 3.5


class GridMap:
    def __init__(self, width=100, height=80, resolution=0.25, origin=(-3.0, -10.0)):
        self.width, self.height = width, height
        self.resolution, self.origin = resolution, np.asarray(origin, float)
        self.observed = np.zeros((height, width), bool)
        self.occupied = np.zeros((height, width), bool)
        self.risk = np.zeros((height, width), float)
        self.slope = np.zeros((height, width), float)
        self.semantic_cost = np.zeros((height, width), float)
        self.last_seen = np.full((height, width), -np.inf)

    def cell(self, xy):
        p = np.floor((np.asarray(xy)[:2] - self.origin) / self.resolution).astype(int)
        return int(p[1]), int(p[0])

    def xy(self, rc):
        r, c = rc
        return self.origin + self.resolution * (np.array([c, r]) + 0.5)

    def inside(self, rc):
        return 0 <= rc[0] < self.height and 0 <= rc[1] < self.width

    def blocked(self, radius, max_slope=None):
        """Inflate occupied cells and map edge by footprint plus caller margin."""
        n = int(math.ceil(radius / self.resolution))
        out = self.occupied.copy()
        if max_slope is not None:
            out |= (self.slope > max_slope)
        # Account for finite source/destination cell extent conservatively.
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                if math.hypot(dr, dc) * self.resolution > radius + self.resolution * 0.71:
                    continue
                ra, rb = max(0, dr), min(self.height, self.height + dr)
                ca, cb = max(0, dc), min(self.width, self.width + dc)
                out[ra:rb, ca:cb] |= self.occupied[ra-dr:rb-dr, ca-dc:cb-dc]
        out[:n, :] = out[-n:, :] = True
        out[:, :n] = out[:, -n:] = True
        return out

    def decay(self, now, max_age=3.5):
        """Decay stale obstacle occupancy to accommodate dynamic obstacles."""
        if not math.isinf(max_age) and max_age > 0:
            stale = (now - self.last_seen > max_age) & self.occupied
            self.occupied[stale] = False

    def footprint_observed(self, xy, radius, now=None, max_age=4.0):
        r, c = self.cell(xy)
        n = int(math.ceil(radius / self.resolution))
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                if math.hypot(dr, dc) * self.resolution > radius:
                    continue
                q = (r + dr, c + dc)
                if not self.inside(q) or not self.observed[q]:
                    return False
                if now is not None and now - self.last_seen[q] > max_age:
                    return False
        return True


def astar(grid, start_xy, goal_xy, cfg, allow_unknown=True, risk_aware=True):
    start, goal = grid.cell(start_xy), grid.cell(goal_xy)
    blocked = grid.blocked(cfg.radius + cfg.margin, max_slope=getattr(cfg, 'max_slope_rad', None))
    if not grid.inside(start) or not grid.inside(goal) or blocked[start] or blocked[goal]:
        return []
    def heuristic(p):
        return math.hypot(p[0]-goal[0], p[1]-goal[1])
    queue, best, previous = [(heuristic(start), 0.0, start)], {start: 0.0}, {}
    while queue:
        _, g, p = heapq.heappop(queue)
        if g > best.get(p, math.inf):
            continue
        if p == goal:
            path = [p]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return [grid.xy(q).tolist() for q in reversed(path)]
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            q = p[0]+dr, p[1]+dc
            if not grid.inside(q) or blocked[q]:
                continue
            if not allow_unknown and not grid.observed[q]:
                continue
            if dr and dc and (blocked[p[0]+dr,p[1]] or blocked[p[0],p[1]+dc]):
                continue
            risk = cfg.risk_weight * grid.risk[q] if risk_aware else 0.0
            slope_cost = getattr(cfg, 'slope_weight', 3.0) * grid.slope[q] if risk_aware else 0.0
            unknown = cfg.unknown_weight if not grid.observed[q] else 0.0
            ng = g + math.hypot(dr,dc) * (1.0 + risk + unknown + slope_cost)
            if ng < best.get(q, math.inf):
                best[q], previous[q] = ng, p
                heapq.heappush(queue, (ng + heuristic(q), ng, q))
    return []


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Navigator:
    def __init__(self, cfg=None, risk_aware=True):
        self.cfg = cfg or Config()
        self.risk_aware = risk_aware
        self.path = []
        self.speed = 0.0
        self.state, self.reason = "READY", "Waiting for observations"

    def stop(self, state, reason):
        self.speed, self.state, self.reason = 0.0, state, reason
        return np.zeros(3)

    def command(self, grid, pose, goal, quality, sensor_age, dt=0.1, now=None):
        cfg = self.cfg
        if not np.all(np.isfinite(pose)) or not np.isfinite(quality) or not np.isfinite(sensor_age):
            return self.stop("HOLD", "Invalid sensor or pose data")
        if sensor_age < 0 or sensor_age > cfg.sensor_timeout:
            return self.stop("HOLD", "Camera data stale")
        if quality < cfg.min_quality:
            return self.stop("HOLD", "Visual tracking unreliable")
        if np.linalg.norm(np.asarray(goal)-np.asarray(pose[:2])) < cfg.goal_tolerance:
            return self.stop("ARRIVED", "Goal tolerance reached")
        if now is not None:
            grid.decay(now, cfg.obstacle_decay_age)
        self.path = astar(grid, pose[:2], goal, cfg, risk_aware=self.risk_aware)
        if len(self.path) < 2:
            return self.stop("BLOCKED", "No feasible route in current map")
        p = np.asarray(pose[:2])
        # Adjacent waypoint avoids cutting a smoothed path through inflated obstacles.
        target = np.asarray(self.path[1])
        heading = math.atan2(*(target-p)[::-1])
        error = wrap(heading-pose[2])
        omega = float(np.clip(2.0*error, -cfg.max_yaw_rate, cfg.max_yaw_rate))
        slow = self.risk_aware and quality < cfg.cautious_quality
        desired = cfg.max_speed * (0.35 if slow else 1.0) * max(0.0, math.cos(error))
        if abs(error) > 0.6:
            desired = 0.0
        speed = min(desired, self.speed + cfg.max_accel * max(0, min(dt, 0.25)))
        blocked = grid.blocked(cfg.radius + cfg.margin, max_slope=getattr(cfg, 'max_slope_rad', None))
        # Swept centre samples out to latency + stopping distance. Unknown support vetoes.
        reach = max(grid.resolution*.5, speed*cfg.sensor_timeout + speed*speed/(2*cfg.braking_accel)) if speed > 0 else 0.0
        for d in np.linspace(0, reach, max(3, int(reach/grid.resolution*3))):
            xy = p + d*np.array([math.cos(pose[2]), math.sin(pose[2])])
            q = grid.cell(xy)
            if not grid.inside(q) or blocked[q]:
                current = grid.cell(p)
                if (grid.inside(current) and not blocked[current] and
                    grid.footprint_observed(p,cfg.radius,now=now,max_age=cfg.support_max_age)
                    and abs(error) > .04):
                    self.speed=0.0
                    self.state,self.reason="TURN","Aligning within verified circular footprint"
                    return np.array([0.0,0.0,omega])
                return self.stop("HOLD", "Obstacle inside stopping envelope")
            if not grid.footprint_observed(xy, cfg.radius, now=now, max_age=cfg.support_max_age):
                return self.stop("OBSERVE", "Ground support unverified")
        self.speed = speed
        self.state = "CAUTIOUS" if slow else "NAVIGATE"
        self.reason = "Reduced speed for degraded vision" if slow else "Following observed route"
        return np.array([speed, 0.0, omega])
