"""Camera-only navigation integration with Visual SLAM and Perception AI."""
import math
import numpy as np
from .navigation import Config, GridMap, Navigator
from .vision import StereoDepth, GroundMapper, camera_mount
from .slam import VisualSLAM
from .perception import PerceptionModel, CLASS_TRAVERSABLE, CLASS_OBSTACLE, CLASS_WATER_MUD, CLASS_VEGETATION


class CameraNavigation:
    def __init__(self, calibration, goal=(10.0, 0.0), initial_base_height=0.50,
                 mount=None, config=None, use_slam=True, use_perception=True):
        self.calibration = calibration
        self.initial_base_height = initial_base_height
        self.goal = np.asarray(goal, float)
        self.grid = GridMap(120, 90, 0.25, (-3.0, -11.0))
        self.mount = camera_mount(height=0.15) if mount is None else mount.copy()
        initial = np.eye(4)
        initial[2, 3] = initial_base_height
        self.stereo = StereoDepth(calibration)

        # SLAM tracking with keyframes and loop closure
        self.use_slam = use_slam
        self.slam = VisualSLAM(calibration, initial @ self.mount)
        self.vo = self.slam.vo  # Compatibility reference

        # Perception AI
        self.use_perception = use_perception
        self.perception = PerceptionModel() if use_perception else None
        self.latest_semantic_mask = None

        self.mapper = GroundMapper(self.grid, calibration)
        self.navigator = Navigator(config or Config(radius=0.50, margin=0.15, max_speed=0.30))
        self.last_stamp = None
        self.bootstrap_done = False
        self.launch_prior_error = None
        self.trajectory = []
        self.map_revision = 0
        self.map_replans = 0

    def process(self, left, right, timestamp, sensor_age=0.0):
        if self.last_stamp is not None and timestamp <= self.last_stamp:
            return self.navigator.stop("HOLD", "Duplicate or reversed camera timestamp"), {"state": "HOLD", "reason": "Camera timestamp invalid"}, None
        dt = 0.10 if self.last_stamp is None else timestamp - self.last_stamp
        self.last_stamp = timestamp

        # 1. Metric stereo depth
        depth = self.stereo.compute(left, right)

        # 2. Visual SLAM motion tracking with keyframe & loop closure
        if self.use_slam:
            T, q, status, slam_diag = self.slam.update(left, depth, timestamp)
        else:
            T, q, status = self.vo.update(left, depth)
            slam_diag = {"slam_enabled": False}
        base = T @ np.linalg.inv(self.mount)
        pose = np.array([base[0, 3], base[1, 3], math.atan2(base[1, 0], base[0, 0])])

        # 3. Perception AI semantic segmentation
        semantic_costs = None
        water_confidence = None
        perception_stats = {}
        if self.use_perception and self.perception is not None:
            sem_grid, conf_grid, sem_mask = self.perception.segment_frame(left, depth, patch_size=40)
            self.latest_semantic_mask = sem_mask
            semantic_costs = self.perception.compute_traversability_cost(conf_grid)
            water_confidence = conf_grid[:,:,CLASS_WATER_MUD]
            total_patches = float(sem_grid.size)
            perception_stats = {
                "traversable_fraction": float(np.sum(sem_grid == CLASS_TRAVERSABLE) / total_patches),
                "obstacle_fraction": float(np.sum(sem_grid == CLASS_OBSTACLE) / total_patches),
                "water_mud_fraction": float(np.sum(sem_grid == CLASS_WATER_MUD) / total_patches),
                "vegetation_fraction": float(np.sum(sem_grid == CLASS_VEGETATION) / total_patches)
            }

        occupied_before = self.grid.occupied.copy()
        # 4. Multi-modal ground and obstacle mapping
        map_stats = {}
        revised = self.use_slam and self.slam.map_revision != self.map_revision
        if revised:
            # Old occupancy is in the old coordinate estimate: discard it.
            self.grid = GridMap(120, 90, 0.25, (-3.0, -11.0))
            self.mapper = GroundMapper(self.grid, self.calibration)
            for kf in self.slam.keyframes:
                kb = kf.T_world_camera @ np.linalg.inv(self.mount)
                # Preserve observation times so historical support can expire.
                self.mapper.update(kf.depth, kf.T_world_camera, kf.timestamp,
                                   expected_ground_z=float(kb[2, 3] - self.initial_base_height))
            self.map_revision = self.slam.map_revision
        if q >= self.navigator.cfg.min_quality:
            map_stats = self.mapper.update(depth, T, timestamp, semantic_cost_map=semantic_costs, water_confidence=water_confidence,
                                           expected_ground_z=float(base[2, 3] - self.initial_base_height))

        if not self.bootstrap_done and map_stats:
            residual = abs(map_stats["ground_z_median"])
            spread = map_stats["ground_z_mad"]
            self.launch_prior_error = residual
            if (map_stats["ground_points"] >= 80 and np.isfinite(residual) and np.isfinite(spread)
                    and residual <= 0.09 and spread <= 0.06):
                rr, cc = np.indices(self.grid.observed.shape)
                x = self.grid.origin[0] + (cc + 0.5) * self.grid.resolution
                y = self.grid.origin[1] + (rr + 0.5) * self.grid.resolution
                pad = (x * x + y * y) <= 1.0 ** 2
                self.grid.observed[pad] = True
                self.grid.last_seen[pad] = timestamp
                self.bootstrap_done = True
            else:
                cmd = self.navigator.stop("HOLD", "Launch-pad stereo check failed")
                self.trajectory.append(pose.tolist())
                return cmd, {
                    "t": timestamp, "pose": pose.tolist(), "quality": q, "vo_status": status,
                    "inliers": self.slam.last_inliers, "state": self.navigator.state,
                    "reason": self.navigator.reason, "command": cmd.tolist(),
                    "valid_depth_fraction": float(np.isfinite(depth).mean()),
                    "launch_prior_checked": True, "launch_prior_error": residual,
                    **slam_diag, **perception_stats, **map_stats
                }, depth

        changed_cells = int(np.count_nonzero(self.grid.occupied != occupied_before))
        if changed_cells: self.map_replans += 1
        map_stats['occupancy_changed_cells'] = changed_cells
        map_stats['map_replans'] = self.map_replans
        # 5. Planning and motion control
        cmd = self.navigator.command(self.grid, pose, self.goal, q, sensor_age, dt, now=timestamp)

        if revised:
            cmd = self.navigator.stop("HOLD", "Map rebuilt after visual pose correction")

        # Pose attitude safety check (allow up to 32 degrees for rough outdoor terrain)
        tilt = math.acos(float(np.clip(base[2, 2], -1.0, 1.0)))
        if tilt > self.navigator.cfg.max_slope_rad:
            cmd = self.navigator.stop("HOLD", "Estimated attitude exceeds safe terrain limit")

        self.trajectory.append(pose.tolist())
        return cmd, {
            "t": timestamp, "pose": pose.tolist(), "quality": q, "vo_status": status,
            "inliers": self.slam.last_inliers, "state": self.navigator.state,
            "reason": self.navigator.reason, "command": cmd.tolist(),
            "valid_depth_fraction": float(np.isfinite(depth).mean()),
            "launch_prior_checked": self.bootstrap_done,
            "launch_prior_error": self.launch_prior_error,
            **slam_diag, **perception_stats, **map_stats
        }, depth
