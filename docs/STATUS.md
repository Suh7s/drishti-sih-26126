# Evidence status

## Established baseline

`results/submission_run` records a successful flat, static forest course: 62.784 simulation seconds, 0.01205 m position RMSE and 0.37826 m conservative clearance. `demo/mission.mp4` contains this actual recorded run with synchronized sensor panels. These results predate the 2.0 integration.

## 2.0 integration

- 41 automated tests pass, including new regressions for stale-obstacle safety, steep-cell inflation, goal integrity, rigid pose corrections, landmark consistency and elevated ground-plane fitting.
- The 40-feature MLP report is in `src/drishti/models/perception_weights.json`: 8,000 procedural image/depth training patches and 2,000 held-out patches; 99.5% held-out accuracy in that narrow synthetic domain.
- The disaster world now contains a collision-enabled ElevationGrid, and evaluation consumes its generated hazard manifest.
- Development terrain runs reached the goal with approximately 2–3 cm RMSE, but failed the full clearance criterion. Their failure summaries are preserved in `results/terrain_attempts`. A follow-up run is validating persistent hazard buffers; no successful rough-terrain mission is asserted yet.

## Not yet demonstrated

Real flood deployment, mud traction, flowing water, rain/night robustness, generalization to real cameras, repeatable moving-obstacle avoidance and a successful full physics loop-closure route. Relocalization and pose-correction unit tests do not replace these experiments.

The runtime is direct Python inside Webots. ROS 2 is not used. The robot is a wheeled rover; historical Spot/Isaac experiments are not the current submission demo.
