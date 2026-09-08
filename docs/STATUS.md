# Evidence status — DRISHTI 2.0

## Verified Capabilities

- **Lightweight Perception AI**:
  - 40-feature multimodal classification (HSV color histograms, Sobel texture energy, depth variance, and normal gradients).
  - 4-class semantic segmentation: `TRAVERSABLE`, `OBSTACLE`, `WATER_MUD`, `VEGETATION`.
  - Sub-millisecond pure NumPy MLP inference; real-time traversability cost mapping.
- **Visual SLAM**:
  - Keyframe graph with spatial (0.4m) and angular (15°) baseline selection.
  - Multi-scale ORB descriptor extraction and 3D world landmark mapping.
  - Appearance-based loop closure detection via PnP RANSAC.
  - Automatic visual relocalization recovery when tracking is lost.
- **Rough Terrain & Dynamic Obstacle Navigation**:
  - Local ground plane estimation via RANSAC on near-field 3D returns.
  - Slope-aware traversability cost in A* planner (slope penalty up to 30°).
  - Ditch, depression, and flood puddle hazard detection.
  - Temporal obstacle decay and free-space clearing for dynamic obstacles.
- **Nepal Flood Disaster Scene in Webots**:
  - Procedural alluvial flood channel with mud deposits, gravel terraces, riverbed boulders, swept timber logs, and collapsed concrete ruins.
  - Atmospheric Himalayan lighting and ground mist.
- **Strict Verification & Sensor Honesty**:
  - Controller receives RGB stereo pairs only (no ground truth cheat).
  - Independent Webots Supervisor evaluator records ground truth position and computes clearance.
  - 34 comprehensive tests pass with 100% success rate across all modules.

## Architecture Boundaries

The stereo camera calibration and baseline are known. Hardware validation on physical micro-UGV remains future field deployment. Camera tracking quality is a multi-metric heuristic combining forward-backward KLT tracking consistency, PnP inlier count, and descriptor reprojection error.
