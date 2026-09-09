# DRISHTI technical brief

## Problem and implemented scope

SIH problem 26126 asks for camera-primary outdoor path detection, GPS-denied
localisation and collision avoidance. This submission implements and demonstrates
a classical stereo navigation baseline in a flat outdoor-style Webots scene.
The robot is a custom four-wheel differential rover, not Spot. No ROS 2 is required.

## Perception and localisation

Two synchronised 640×400 RGB cameras use a 0.16 m baseline and 1.4 rad horizontal
field of view. StereoSGBM estimates disparity; left/right consistency rejects
unreliable matches. Unprojection uses the known intrinsics and baseline.

Good-feature detection and forward/backward Lucas–Kanade flow establish temporal
correspondences. RANSAC PnP estimates metric motion; low-quality tracking stops
commands rather than silently resetting the global frame. There is no loop closure.

The mapper projects points into a 0.25 m grid. Near-plane support and repeated
height evidence update traversability and hazards. A bottom-central image region
checks the declared launch plane without selecting only plane-consistent points.
This ground model assumes flat terrain and does not establish ditch or slope safety.

## Planning and control

A* penalises risk and unknown cells, inflates obstacles, and rejects diagonal
corner cutting. A circular planning footprint covers the wheel envelope with
additional margin. The local supervisor checks tracking, image age, ground support,
and a stopping envelope before applying differential wheel velocities.

Ground support persists for this static course. Moving-obstacle clearing and
freshness-aware dynamic occupancy are not implemented. Pose quality is a heuristic,
not a calibrated uncertainty probability or formal safety guarantee.

## Evaluation separation

The rover uses the normal Webots Robot API and camera devices. A separate Supervisor
records actual position and controls the cinematic viewpoint. Its pose is never
passed to the navigation controller. The evaluator compares trajectories without
post-hoc alignment. Clearance uses a 0.42 m enclosing circle against the two declared
obstacle boxes; successful scoring requires at least 0.05 m clearance and arrival
within 0.5 m of the goal. Decorative scene geometry is not included in that metric.

A known start pose is a setup assumption. The launcher regenerates the course to
avoid starting from an accidentally saved end pose, and the evaluator rejects an
incorrect starting translation.

## Reproducibility

`python webots/launch.py --record --exit` creates a separate run directory with
configuration, RGB/depth diagnostics, navigation logs, independent truth logs,
recordings and evaluation JSON. `--fault blackout` optionally injects black RGB
frames at t=12–14 s to test stopping and recovery. Refer to saved evidence before
claiming that optional test passed.

The final main movie has a visual overlay added by `visuals/package_demo.py`.
Its source imagery and telemetry are from the same recorded run. The concluding
card reports that run's measurements. `demo/index.html` provides the standalone offline
viewer; `visuals/build_assets.py` regenerates the repository's trajectory graphics.

## Next research steps

A stronger research system would evaluate a defined hypothesis across many terrain,
lighting and sensor-failure conditions, comparing a baseline with an uncertainty-aware
method. Priorities are terrain-relative ground estimation, slope and negative-obstacle
reasoning, loop closure, dynamic occupancy, and controlled hardware validation.
A learned semantic model should be evaluated on held-out scenes before claiming
semantic understanding. The present run does not establish flood-response suitability.
