"""Rectified stereo, metric visual odometry and observed-ground mapping.

Classical baseline, not trained semantic AI or loop-closing SLAM. Tracking quality
is a diagnostic heuristic, not a calibrated probability. Lost VO never silently
reinitializes the global frame. Use reset only when explicitly beginning a new run.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .navigation import GridMap


@dataclass
class Calibration:
    fx: float = 420.0
    fy: float = 420.0
    cx: float = 320.0
    cy: float = 240.0
    baseline: float = 0.12
    width: int = 640
    height: int = 480

    @property
    def K(self):
        return np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]], np.float64)


def gray(image):
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)


class StereoDepth:
    def __init__(self, calib):
        if calib.baseline <= 0 or calib.fx <= 0:
            raise ValueError("Positive metric baseline and focal length required")
        self.calib = calib
        common = dict(numDisparities=96, blockSize=5, P1=8*25, P2=32*25,
                      uniquenessRatio=10, speckleWindowSize=60, speckleRange=2,
                      disp12MaxDiff=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left = cv2.StereoSGBM_create(minDisparity=0, **common)
        self.right = cv2.StereoSGBM_create(minDisparity=-96, **common)

    def compute(self, left, right):
        l, r = gray(left), gray(right)
        if l.shape != r.shape or l.shape != (self.calib.height,self.calib.width):
            raise ValueError("Images must match calibration and be rectified/synchronized")
        d = self.left.compute(l,r).astype(np.float32)/16.0
        rd = self.right.compute(r,l).astype(np.float32)/16.0
        yy, xx = np.indices(d.shape)
        rx = np.rint(xx-d).astype(int)
        in_frame = (rx >= 0) & (rx < d.shape[1])
        rd_match = rd[yy, np.clip(rx,0,d.shape[1]-1)]
        consistent = np.abs(d+rd_match) < 2.5
        depth = np.full(d.shape, np.nan, np.float32)
        valid = (d > 0.75) & (d < 95) & in_frame & consistent & (rd_match > -96)
        depth[valid] = self.calib.fx*self.calib.baseline/d[valid]
        depth[(depth < 0.25) | (depth > 10)] = np.nan
        return depth


def unproject(uv, z, c):
    return np.column_stack(((uv[:,0]-c.cx)*z/c.fx, (uv[:,1]-c.cy)*z/c.fy, z))


def estimate_motion(object_points, image_points, K):
    """Return previous-camera to current-camera transform and inlier diagnostics."""
    if len(object_points) < 12:
        return None, 0, 0.0
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(object_points,np.float32), np.asarray(image_points,np.float32),
        K, None, iterationsCount=150, reprojectionError=2.5, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    count = 0 if inliers is None else len(inliers)
    ratio = count / max(1,len(object_points))
    if not ok or count < 12 or ratio < 0.35:
        return None, count, ratio
    idx = inliers.ravel()
    rvec,tvec = cv2.solvePnPRefineLM(np.asarray(object_points,np.float32)[idx],
                                   np.asarray(image_points,np.float32)[idx],K,None,rvec,tvec)
    T = np.eye(4)
    T[:3,:3], T[:3,3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T, count, ratio


class VisualOdometry:
    def __init__(self, calib, initial_T_world_camera=None):
        self.calib = calib
        self.T = np.eye(4) if initial_T_world_camera is None else initial_T_world_camera.copy()
        self.anchor_gray, self.anchor_depth, self.anchor_points = None,None,None
        self.failures = 0
        self.quality = 0.0
        self.last_inliers = 0

    def _anchor(self, g, depth):
        mask = (np.isfinite(depth).astype(np.uint8)*255)
        pts = cv2.goodFeaturesToTrack(g, maxCorners=900, qualityLevel=0.003,
                                    minDistance=7, mask=mask, blockSize=7)
        self.anchor_gray, self.anchor_depth, self.anchor_points = g.copy(), depth.copy(), pts
        return 0 if pts is None else len(pts)

    def update(self, image, depth):
        g = gray(image)
        if self.anchor_gray is None or self.anchor_points is None:
            n = self._anchor(g,depth)
            self.quality = min(0.8,n/250) if n >= 30 else 0.0
            return self.T.copy(), self.quality, "INITIALIZED" if n >= 30 else "NO_FEATURES"
        p0 = self.anchor_points
        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.anchor_gray,g,p0,None,
                                               winSize=(21,21),maxLevel=3)
        if p1 is None:
            self.quality = 0.0
            return self.T.copy(),0.0,"LOST"
        back, bs, _ = cv2.calcOpticalFlowPyrLK(g,self.anchor_gray,p1,None,
                                            winSize=(21,21),maxLevel=3)
        if back is None or bs is None:
            self.quality = 0.0
            return self.T.copy(), 0.0, "LOST"
        uv0, uv1 = p0.reshape(-1,2),p1.reshape(-1,2)
        valid = (status.ravel()==1)&(bs.ravel()==1)&(np.linalg.norm(back.reshape(-1,2)-uv0,axis=1)<1.0)
        px = np.rint(uv0).astype(int)
        z = self.anchor_depth[np.clip(px[:,1],0,g.shape[0]-1),np.clip(px[:,0],0,g.shape[1]-1)]
        valid &= np.isfinite(z)&(uv1[:,0]>=0)&(uv1[:,0]<g.shape[1])&(uv1[:,1]>=0)&(uv1[:,1]<g.shape[0])
        motion, n, ratio = estimate_motion(unproject(uv0[valid],z[valid],self.calib),uv1[valid],self.calib.K)
        self.last_inliers = n
        # Bound jumps between accepted frames. Failure retains last good anchor and pose.
        if motion is None or np.linalg.norm(motion[:3,3]) > 0.5:
            self.failures += 1
            self.quality = 0.0
            return self.T.copy(),0.0,"LOST"
        self.T = self.T @ np.linalg.inv(motion)
        self.quality = float(min(1.0,n/100)*ratio)
        self.failures = 0
        self._anchor(g,depth)
        return self.T.copy(),self.quality,"TRACKING"


class GroundMapper:
    def __init__(self, grid, calibration, ground_z=0.0):
        self.grid,self.c,self.ground_z = grid,calibration,ground_z
        self.hazard_votes=np.zeros(grid.occupied.shape,np.uint8)

    def update(self, depth, T_world_camera, timestamp):
        """Flat-ground baseline. Do not deploy unchanged on slopes or steps.

        Observed support requires >=3 independent pixel samples near the known
        start ground plane. Elevated and below-plane returns mark hazards.
        There is no ray-carving of free support and no hole filling across gaps.
        """
        rows,cols = np.indices(depth.shape)
        mask = np.isfinite(depth)&(depth<7.0)
        mask &= (rows%3==0)&(cols%3==0)
        uv = np.column_stack((cols[mask],rows[mask]))
        points = unproject(uv,depth[mask],self.c)
        world = np.einsum('ij,kj->ik',points,T_world_camera[:3,:3])+T_world_camera[:3,3]
        cells = np.floor((world[:,:2]-self.grid.origin)/self.grid.resolution).astype(int)
        valid = (cells[:,0]>=0)&(cells[:,0]<self.grid.width)&(cells[:,1]>=0)&(cells[:,1]<self.grid.height)
        cells,world,uv = cells[valid],world[valid],uv[valid]
        shape = self.grid.occupied.shape
        support,obstacle = np.zeros(shape,int),np.zeros(shape,int)
        z = world[:,2]-self.ground_z
        ground = np.abs(z)<0.10
        hazard = ((z>0.16)&(z<1.3))|(z < -0.15)
        np.add.at(support,(cells[ground,1],cells[ground,0]),1)
        np.add.at(obstacle,(cells[hazard,1],cells[hazard,0]),1)
        # Reject isolated disparity outliers. A hazard needs repeated majority
        # evidence in its cell. This is a heuristic, evaluated only on flat scenes.
        total=np.zeros(shape,int)
        np.add.at(total,(cells[:,1],cells[:,0]),1)
        seen = (support>=3)&(support>=0.6*total)
        candidate=(obstacle>=5)&(obstacle>=0.65*total)
        self.hazard_votes[candidate]=np.minimum(self.hazard_votes[candidate].astype(int)+1,3)
        self.hazard_votes[seen]=0
        occ=candidate&(self.hazard_votes>=2)
        self.grid.observed |= seen|occ
        self.grid.last_seen[seen|occ] = timestamp
        # Persistent obstacle evidence is conservative; dynamic clearing is future work.
        self.grid.occupied |= occ
        self.grid.risk[seen] = np.clip(1.0-support[seen]/25,0,1)*0.25
        # Select the diagnostic region in image coordinates, without filtering
        # for agreement with the assumed plane (which would make this circular).
        launch_region=(uv[:,1]>.55*self.c.height)&(np.abs(uv[:,0]-self.c.cx)<.3*self.c.width)
        ground_z = z[launch_region]
        # This is an image-derived check of the declared flat launch-pad prior.
        # It is diagnostic only: it never changes the assumed plane or VO pose.
        return {"support_cells":int(seen.sum()),"hazard_cells":int(occ.sum()),
                "ground_points":int(ground_z.size),
                "ground_z_median":float(np.median(ground_z)) if ground_z.size else float("nan"),
                "ground_z_mad":float(np.median(np.abs(ground_z-np.median(ground_z)))) if ground_z.size else float("nan")}


def _camera_pitch(pitch_degrees):
    a=np.deg2rad(pitch_degrees)
    return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])


def camera_mount(height=0.65, forward=0.32, left=0.06, pitch_degrees=45.0):
    """OpenCV optical frame (right, down, forward) to Spot body (forward, left, up).

    This transform belongs to stereo unprojection and PnP. It must not be
    passed directly to Isaac's Camera with ``camera_axes='ros'``.
    """
    optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    T=np.eye(4)
    T[:3,:3]=_camera_pitch(pitch_degrees)@optical
    T[:3,3]=[forward,left,height]
    return T


def camera_ros_mount(height=0.65, forward=0.32, left=0.06, pitch_degrees=45.0):
    """Isaac legacy Camera ``ros`` axes to Spot body.

    That API defines +Y up and +Z forward, so its local basis is (left, up,
    forward). Its +Z viewing ray therefore agrees with OpenCV optical +Z.
    """
    T=np.eye(4)
    ros_from_optical=np.diag([-1.0,-1.0,1.0])
    T[:3,:3]=camera_mount(0,0,0,pitch_degrees)[:3,:3]@ros_from_optical
    T[:3,3]=[forward,left,height]
    return T
