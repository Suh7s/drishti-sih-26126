# Evidence status

## Established baseline

`results/submission_run` records a successful flat, static forest course: 62.784 simulation seconds, 0.01205 m position RMSE and 0.37826 m conservative clearance. `demo/mission.mp4` contains this actual recorded run with synchronized sensor panels. These results predate the 2.0 integration.

## 2.0 integration

- 41 automated tests pass, including new regressions for stale-obstacle safety, steep-cell inflation, goal integrity, rigid pose corrections, landmark consistency and elevated ground-plane fitting.
- The 40-feature MLP report is in `src/drishti/models/perception_weights.json`: 8,000 procedural image/depth training patches and 2,000 held-out patches; 99.5% held-out accuracy in that narrow synthetic domain.
- The disaster world contains a collision-enabled ElevationGrid, and evaluation consumes its generated hazard manifest.
- `results/disaster_validation_06` validates the rough Himalayan flood terrain mission: 36.864 simulation seconds, 0.02652 m (2.65 cm) position RMSE, 99.74% tracking continuity, 7.49° max body tilt, and verified arrival at the 10 m disaster relief goal. `demo/index.html` and `demo/mission.mp4` display this measured disaster run with synchronized multi-view video and real-time telemetry. Historical development runs are preserved in `results/terrain_attempts`.

## Not yet demonstrated

Real flood deployment, mud traction, flowing water, rain/night robustness, generalization to real cameras, repeatable moving-obstacle avoidance and a successful full physics loop-closure route. Relocalization and pose-correction unit tests do not replace these experiments.

The runtime is direct Python inside Webots. ROS 2 is not used. The robot is a wheeled rover; historical Spot/Isaac experiments are not the current submission demo.
