Continue this existing project on Ubuntu. Implement, debug and verify it in Isaac
Sim 6.0.1 on my RTX 5080 machine with 32 GB RAM. The SIH submission is tomorrow.
I want a working camera-first outdoor navigation demonstration with Spot and a
polished cinematic recording. I will make the slides myself. No physical hardware
is available. Work autonomously within this project and preserve run evidence.

Read START_HERE.md and docs/STATUS.md first. This package was developed on a Mac.
Its Isaac adapter has never been executed. Do not assume it works because it
compiles. Inspect the actual installed 6.0.1 example and APIs before changing the
integration. Use the installed Spot locomotion policy and assets. Do not spend
the deadline training a new locomotion policy.

Work in this order:

1. Locate Isaac Sim 6.0.1 and check its Python, OpenCV availability, GPU and driver.
   Run its own Spot standalone example to establish that the asset and policy work.
2. Run our portable tests. Fix API imports, initialization, callback timing, camera
   mounting to the moving rigid body, synchronization and frame conventions.
3. Verify stationary stereo with capture mode. Our depth uses SGBM and our pose
   uses PnP visual odometry. Neither uses simulator truth. The initial launch-pad
   pose/height and flat support patch are declared priors that need checking.
4. Address mapping weaknesses before navigation. An analytic RGB integration test
   found false obstacles from stereo outliers. Majority/repeated evidence filtering
   was added, but the subsequent closed-loop test stopped at unobserved footprint
   cells after about 0.47 m. Diagnose calibration, ground uncertainty and field of
   view properly. Do not solve this by marking unseen terrain free or feeding truth
   geometry/pose to the navigation module. The camera facing/coverage and footprint
   support requirement need particular attention.
5. Get repeated obstacle-free two-metre camera-based navigation working on a flat
   textured launch course, then add an obstacle and a controlled camera failure.
   Spot must physically walk. Preserve stops and failures in the logs. Check the
   conservative circular footprint against the actual leg/body swept envelope.
6. Add richer outdoor materials and natural-looking assets only after functionality
   works. Keep a flat traversable course compatible with the supplied policy.
   Improve the current primitive scenery if local/vendor assets are available.
   Do not claim rough-terrain or ditch competence from the flat policy.
7. Verify cinematic.py: smooth establishing-to-following camera, warm lighting,
   independent camera render output, readable minimal overlays and smooth playback.
   Navigation uses only the onboard stereo pair. Simulator truth may drive the
   presentation camera and evaluator, never the navigation state estimate.
8. Record a continuous actual run to MP4, camera/depth evidence and telemetry. Check
   frame rate against simulation timestamps, inspect the video and measure completion,
   collisions/falls, tracking failures and latency. Save raw uncut footage too.
9. End with the exact successful launch command, verified output files, measured
   results and limitations. Update docs/STATUS.md with evidence as work progresses.

Scope boundaries: this is a classical vision baseline, not trained Perception AI,
loop-closing SLAM, Nav2 MPPI, calibrated uncertainty or active viewpoint recovery.
Those were research directions from the original proposal and are not implemented.
The existing browser replay and CSVs test a 2-D planner using synthetic observations
and exact pose. They are not camera navigation benchmarks or Isaac recordings.
If a truth-assisted diagnostic mode becomes useful, keep it separate and visibly
label it. Never relabel it as camera-only autonomy.

Do not build a presentation. Prioritize repeatable real simulation behavior, then
cinematic polish. Fix problems instead of presenting untested scripts as complete.
