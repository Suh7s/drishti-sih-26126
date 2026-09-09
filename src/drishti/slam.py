"""Keyframe-based Visual SLAM with Loop Closure and Relocalization.

Extends metric stereo visual odometry with:
- Spatially and temporally spaced keyframe storage
- Multi-scale ORB feature extraction and descriptor matching
- Appearance-based loop closure detection via PnP RANSAC
- Bounded SE(3) pose-chain loop error distribution (not bundle adjustment)
- Global relocalization from lost tracking
Compatible with the CameraNavigation pipeline.
"""
from dataclasses import dataclass
import math
import numpy as np
import cv2
from .vision import VisualOdometry, unproject, estimate_motion


@dataclass
class Keyframe:
    id: int
    timestamp: float
    T_world_camera: np.ndarray
    gray: np.ndarray
    depth: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    points_3d: np.ndarray  # (N, 3) in world coordinates


class VisualSLAM:
    """Visual SLAM managing local VO, keyframe memory, loop closure, and relocalization."""

    def __init__(self, calib, initial_T_world_camera=None,
                 min_keyframe_dist=0.40, min_keyframe_angle_deg=15.0):
        self.calib = calib
        self.vo = VisualOdometry(calib, initial_T_world_camera)
        self.min_dist = min_keyframe_dist
        self.min_angle_rad = math.radians(min_keyframe_angle_deg)

        self.keyframes = []
        self.next_keyframe_id = 0
        self.last_keyframe_pose = self.vo.T.copy()
        self.loop_closures = 0
        self.relocalizations = 0
        self.last_loop_info = None
        self.frame_count = 0
        self.map_revision = 0
        self.last_loop_frame = -1000

        # ORB detector for invariant keyframe matching and loop closure
        self.orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=4, edgeThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @property
    def T(self):
        return self.vo.T

    @property
    def quality(self):
        return self.vo.quality

    @property
    def failures(self):
        return self.vo.failures

    @property
    def last_inliers(self):
        return self.vo.last_inliers

    def _should_create_keyframe(self, current_T, quality, inliers):
        if len(self.keyframes) == 0:
            return True

        delta_T = np.linalg.inv(self.last_keyframe_pose) @ current_T
        dist = float(np.linalg.norm(delta_T[:3, 3]))
        R_delta = delta_T[:3, :3]
        cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(math.acos(cos_angle))

        # Create keyframe on spatial/rotational baseline, or when tracking inliers drop
        if dist >= self.min_dist or angle >= self.min_angle_rad:
            return True
        if inliers < 45 and quality > 0.35 and dist >= (self.min_dist * 0.5):
            return True

        return False

    def _extract_keyframe_data(self, gray, depth, T_world_camera, timestamp):
        kp, des = self.orb.detectAndCompute(gray, None)
        if kp is None or len(kp) < 15 or des is None:
            return None

        # Extract metric 3D points in world coordinates for each keypoint
        pts_2d = np.array([p.pt for p in kp], dtype=np.float32)
        px = np.rint(pts_2d).astype(int)
        valid_px = (px[:, 0] >= 0) & (px[:, 0] < depth.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < depth.shape[0])

        z_vals = np.full(len(kp), np.nan, dtype=np.float32)
        valid_idx = np.flatnonzero(valid_px)
        z_vals[valid_idx] = depth[px[valid_idx, 1], px[valid_idx, 0]]

        has_depth = np.isfinite(z_vals) & (z_vals > 0.25) & (z_vals < 9.0)
        if np.sum(has_depth) < 12:
            return None

        # Camera frame to world frame
        cam_3d = unproject(pts_2d[has_depth], z_vals[has_depth], self.calib)
        world_3d = np.einsum('ij,kj->ik', cam_3d, T_world_camera[:3, :3]) + T_world_camera[:3, 3]

        # Filter down to keypoints and descriptors with valid 3D points
        filtered_kp = [kp[i] for i in np.flatnonzero(has_depth)]
        filtered_des = des[has_depth]

        kf = Keyframe(
            id=self.next_keyframe_id,
            timestamp=timestamp if timestamp is not None else 0.0,
            T_world_camera=T_world_camera.copy(),
            gray=gray.copy(),
            depth=depth.copy(),
            keypoints=filtered_kp,
            descriptors=filtered_des,
            points_3d=world_3d
        )
        self.next_keyframe_id += 1
        return kf

    def detect_loop_closure(self, current_gray, current_T, max_search_dist=1.0):
        """Match current features against past non-adjacent keyframes."""
        if len(self.keyframes) < 4:
            return None, None

        curr_kp, curr_des = self.orb.detectAndCompute(current_gray, None)
        if curr_kp is None or curr_des is None or len(curr_des) < 20:
            return None, None

        curr_pos = current_T[:3, 3]

        best_kf = None
        best_inliers = 0
        best_T_kf_curr = None

        # Examine keyframes excluding the most recent 3
        for kf in self.keyframes[:-3]:
            kf_pos = kf.T_world_camera[:3, 3]
            dist = float(np.linalg.norm(curr_pos - kf_pos))
            positions = [f.T_world_camera[:3, 3] for f in self.keyframes if f.id >= kf.id] + [curr_pos]
            travel = sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:]))
            if dist > max_search_dist or travel < 3.0:
                continue

            # Match ORB descriptors using ratio test
            matches = self.matcher.knnMatch(curr_des, kf.descriptors, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                    good_matches.append(m_n[0])

            if len(good_matches) < 15:
                continue

            # 3D points from keyframe, 2D points from current frame
            pts_3d = np.array([kf.points_3d[m.trainIdx] for m in good_matches], dtype=np.float32)
            pts_2d = np.array([curr_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32)

            # Solve PnP directly in world frame to estimate current camera pose
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.calib.K, None,
                iterationsCount=150, reprojectionError=3.0, confidence=0.99
            )

            if ok and inliers is not None and len(inliers) >= 20 and len(inliers) / len(good_matches) >= .55:
                num_inl = len(inliers)
                if num_inl > best_inliers:
                    best_inliers = num_inl
                    best_kf = kf

                    R, _ = cv2.Rodrigues(rvec)
                    # Pose of world wrt current camera: p_cam = R @ p_world + t
                    # Invert to get T_world_camera:
                    T_cam_world = np.eye(4)
                    T_cam_world[:3, :3] = R
                    T_cam_world[:3, 3] = tvec.ravel()
                    best_T_world_curr = np.linalg.inv(T_cam_world)
                    best_T_kf_curr = best_T_world_curr

        if best_kf is not None:
            return best_kf, best_T_kf_curr

        return None, None

    def attempt_relocalization(self, current_gray):
        """Relocalize camera when VO tracking is lost."""
        if len(self.keyframes) == 0:
            return None

        curr_kp, curr_des = self.orb.detectAndCompute(current_gray, None)
        if curr_kp is None or curr_des is None or len(curr_des) < 25:
            return None

        best_kf = None
        best_inliers = 0
        best_pose = None

        for kf in reversed(self.keyframes):
            matches = self.matcher.knnMatch(curr_des, kf.descriptors, k=2)
            good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance]

            if len(good_matches) < 16:
                continue

            pts_3d = np.array([kf.points_3d[m.trainIdx] for m in good_matches], dtype=np.float32)
            pts_2d = np.array([curr_kp[m.queryIdx].pt for m in good_matches], dtype=np.float32)

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.calib.K, None,
                iterationsCount=200, reprojectionError=2.5, confidence=0.999
            )

            if ok and inliers is not None and len(inliers) >= 20 and len(inliers) / len(good_matches) >= .55:
                if len(inliers) > best_inliers:
                    best_inliers = len(inliers)
                    best_kf = kf
                    R, _ = cv2.Rodrigues(rvec)
                    T_cam_world = np.eye(4)
                    T_cam_world[:3, :3] = R
                    T_cam_world[:3, 3] = tvec.ravel()
                    best_pose = np.linalg.inv(T_cam_world)

        if best_kf is not None and best_inliers >= 16:
            self.relocalizations += 1
            # Re-anchor VO at verified relocalized pose
            self.vo.T = best_pose.copy()
            self.vo.anchor_gray = None
            self.vo.anchor_points = None
            self.vo.failures = 0
            self.vo.quality = .75
            self.vo.last_inliers = best_inliers
            self.map_revision += 1
            return best_pose

        return None

    def correct_pose_chain(self, matched_kf, current_T, corrected_T):
        """Distribute a bounded loop correction with proper rotations.

        This is a pose-chain approximation, not a least-squares pose graph.
        Landmark coordinates move with their owning keyframe. Mapping clients
        must rebuild their map when map_revision changes.
        """
        delta = corrected_T @ np.linalg.inv(current_T)
        rotation_vector = cv2.Rodrigues(delta[:3, :3])[0]
        span = max(1, self.next_keyframe_id - matched_kf.id)
        for kf in self.keyframes:
            alpha = np.clip((kf.id - matched_kf.id) / span, 0, 1)
            correction = np.eye(4)
            correction[:3, :3] = cv2.Rodrigues(rotation_vector * alpha)[0]
            correction[:3, 3] = delta[:3, 3] * alpha
            kf.T_world_camera = correction @ kf.T_world_camera
            kf.points_3d = np.einsum('ij,kj->ik', kf.points_3d, correction[:3, :3]) + correction[:3, 3]
        if self.keyframes:
            self.last_keyframe_pose = self.keyframes[-1].T_world_camera.copy()
        self.vo.T = corrected_T.copy()
        self.map_revision += 1
        return corrected_T.copy()

    def update(self, image, depth, timestamp=None):
        """Process stereo image pair frame and update SLAM state.

        Returns:
            T: 4x4 transform T_world_camera
            quality: float [0, 1] tracking quality
            status: "TRACKING", "LOOP_CLOSED", "RELOCALIZED", or "LOST"
            diagnostics: dict with SLAM metrics
        """
        gray = self.vo.calib and cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        self.frame_count += 1
        T, q, status = self.vo.update(image, depth)

        loop_closed = False
        relocalized = False
        keyframe_added = False

        if status == "LOST" or q < 0.25:
            # Attempt visual relocalization
            reloc_pose = self.attempt_relocalization(gray)
            if reloc_pose is not None:
                T = reloc_pose
                self.vo._anchor(gray, depth)
                q = 0.75
                status = "RELOCALIZED"
                relocalized = True
        elif (status == "TRACKING" and q >= 0.35) or (len(self.keyframes) == 0 and status in ("INITIALIZED", "TRACKING") and q >= 0.25):
            # Check for loop closures every few frames if enough keyframes exist
            if len(self.keyframes) >= 4 and self.frame_count % 30 == 0 and self.frame_count - self.last_loop_frame > 150:
                matched_kf, corrected_T = self.detect_loop_closure(gray, T)
                if matched_kf is not None and corrected_T is not None:
                    # Smoothly blend / correct drift towards loop closure pose
                    drift = np.linalg.norm(corrected_T[:3, 3] - T[:3, 3])
                    angle = np.linalg.norm(cv2.Rodrigues(corrected_T[:3, :3] @ T[:3, :3].T)[0])
                    if drift < .6 and angle < math.radians(15):
                        T = self.correct_pose_chain(matched_kf, T, corrected_T)
                        self.last_loop_frame = self.frame_count
                        self.loop_closures += 1
                        loop_closed = True
                        status = "LOOP_CLOSED"
                        self.last_loop_info = {
                            "kf_id": matched_kf.id,
                            "drift_m": float(drift),
                            "timestamp": timestamp
                        }

            # Check if current frame qualifies as a new Keyframe
            if self._should_create_keyframe(T, q, self.vo.last_inliers):
                kf = self._extract_keyframe_data(gray, depth, T, timestamp)
                if kf is not None:
                    self.keyframes.append(kf)
                    self.last_keyframe_pose = T.copy()
                    keyframe_added = True

        diagnostics = {
            "map_revision": self.map_revision,
            "slam_status": status,
            "keyframes_count": len(self.keyframes),
            "loop_closures_count": self.loop_closures,
            "relocalizations_count": self.relocalizations,
            "keyframe_added": keyframe_added,
            "loop_closed": loop_closed,
            "relocalized": relocalized
        }

        return T, q, status, diagnostics
