# Ubuntu verification, 2026-09-08 (in progress)

- Isaac Sim 6.0.1 is at `/home/s/isaacsim`; bundled Python 3.12.13, NumPy 1.26.4 and OpenCV 4.13.0. Host: Ubuntu 24.04.4, RTX 5080, driver 595.84, CUDA 13.2, 32 GB RAM.
- All 19 portable tests pass after camera-frame, launch-prior, RGB conversion, stereo, and material changes.
- NVIDIA's unmodified Spot test initialized the GPU pipeline but was killed while loading its large grid scene; this is an environment failure, not a walking pass.
- Adapter fixes verified against installed source: sensor rates divide renderer rate, Camera initialization follows playback, RGB annotators return float LDR data, and legacy ROS cameras use +Z as viewing ray. Cameras are parented to `/World/Spot/body`.
- `results/isaac_capture_004` contains the first clean adapter run: 34 synchronized telemetry rows, but all-black RGB from the conversion bug.
- `results/isaac_capture_007` contains the first corrected rendered capture: 51 rows, valid SGBM depth 23.6--25.5% (median 1.26 m), but VO stayed `LOST`; Spot commanded zero.
- `results/isaac_capture_009` and `results/isaac_capture_010` preserve material-backed/brighter attempts. RGB is nonzero, but valid disparity is roughly 2--7% and VO remains `LOST`; the safety gate holds Spot.
- The required camera-only two-metre walking test has not passed. No autonomous or collision-free Isaac navigation claim is made. The installed flat-terrain Spot policy and asset are present, but an actual walking assertion remains pending.
- `results/isaac_cinematic_warmup_001` preserves a separate presentation-camera warm-up from Isaac assets: `cinematic.mp4` is 1280x720, 20 fps, 82 frames; `camera_preview.mp4` is 1280x480, 20 fps, 34 frames. It is a stationary/hold capture and must not be labeled an autonomous navigation movie.

# Implementation and evidence status at Mac handoff

## Implemented and locally exercised

- A* planning, risk costs, obstacle inflation and conservative motion supervision.
- Stop conditions for stale timestamps, missing support and low tracking quality.
- StereoSGBM with left/right consistency and metric calibrated depth.
- Optical flow with forward/backward checks and RANSAC PnP visual odometry.
- Flat-ground support mapping and RGB-to-command integration.
- Analytic RGB renderer for testing stereo and visual motion estimation.
- Four synthetic 2-D planning scenarios with baseline and risk-aware configurations.
- Offline replay viewer with scenario/mode selection and time controls.

18 portable tests passed after the final ground-filter change, including a run
with Python warnings treated as errors before the GitHub push. The change added
repeated majority hazard evidence. A follow-up closed-loop analytic RGB experiment
advanced approximately 0.47 m and stopped at unobserved footprint support. Fix
coverage/mapping on Ubuntu. This is not an autonomous camera navigation success.

24 synthetic planning executions completed with ARRIVED in the included CSV.
They use exact pose and idealized 360-degree ray observations with occlusion.
They have no legged dynamics, image inference or SLAM. The uncertain-shortcut layout
is identical across the three seeds, so those repeats are not independent trials.
Both controllers share the safety supervisor. Do not present completion counts as
an outdoor navigation success rate or claim statistical significance.

For seed 7 on the scripted uncertain shortcut, baseline duration was 33.30 seconds
and risk-aware duration was 42.75 seconds. Scripted risk exposure was 12.555 seconds
and zero respectively. This demonstrates the configured objective's trade-off,
not learned uncertainty or generalization.

The replay was visually inspected in the Codex browser at desktop width. The
slider changed the displayed trajectory/state. Full automated browser checks did
not run because launching an external headless Chrome was blocked in the local
environment. Do not claim complete browser QA.

## Written, not executed in Isaac Sim

- Isaac Sim 6.0.1 Spot scene launcher.
- Stereo attachment to a detected body rigid prim.
- Installed NVIDIA flat-terrain policy control.
- Camera/depth preview video and JSONL telemetry capture.
- Cinematic camera and 1280x720 MP4 recording.

The 6.0.1 adapter uses the documented legacy Camera facade and SimulationManager
callback path. These need comparison with the user's installed files. Potential
runtime issues include extension availability, initialization order, camera frame
metadata, pose tensor conversion and rigid-body parent identification.

## Important remaining functional work

1. Confirm nominal settled body height and camera extrinsics. The mapper assumes
   a flat ground plane at world z=0 and launch height 0.50 m. Errors can create
   spurious terrain hazards.
2. Resolve unobserved support around the footprint without falsely filling holes.
3. Verify actual Spot stability and conservative speed/footprint limits.
4. Check movement, arrival and collision evidence in the physics simulation.
5. Validate movie recording frame counts and simulation-time cadence.
6. Ensure resources close and videos finalize on errors/interruption as well as
   successful exit. Current cleanup closes the app but recording cleanup on an
   exception needs hardening.

## Not implemented

Learned semantic perception, trained traversability AI, loop closure, long-distance
SLAM, IMU fusion, Nav2/ROS integration, MPPI, moving-obstacle prediction, automatic
viewpoint recovery, calibrated risk probabilities, real-world testing and rough
terrain locomotion training. Persistent obstacle evidence also lacks dynamic
clearing. A hold can remain a hold indefinitely.

## Sources consulted

- https://docs.isaacsim.omniverse.nvidia.com/6.0.1/py/source/extensions/isaacsim.core.simulation_manager/config/python_api.html
- https://docs.isaacsim.omniverse.nvidia.com/6.0.1/py/source/deprecated/isaacsim.sensors.camera/config/python_api.html
- https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_camera.html
- https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_simulation/ext_isaacsim_robot_policy_example.html
- https://github.com/isaac-sim/IsaacSim/blob/main/source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

External libraries and installed NVIDIA assets retain their own licenses. The
package does not redistribute Spot assets, policy weights or NVIDIA example code.
