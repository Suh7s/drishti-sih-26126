# Evidence status

## Verified

- Webots R2025a runs on the development Mac M4 Pro.
- The rover completes a closed-loop route using stereo RGB, metric visual odometry,
  observed-ground mapping and wheel commands.
- A separate Supervisor logs simulator position for evaluation; it never sends
  positions or maps into the rover controller.
- The final detailed forest run reaches the goal with 37.83 cm conservative
  clearance, 1.21 cm position RMSE, and 62.784 seconds of simulated navigation.
- 23 portable tests pass, including regression checks for turning, stale images,
  tracking loss, and an independent launch-plane diagnostic.

Raw logs and scores: `results/submission_run/`. An earlier simpler course remains
in `results/validated_webots/`. These are controlled demonstration runs, not a
statistical evaluation of outdoor robustness.

## Boundaries

The start pose, stereo calibration, mount and launch pad are known. The current
mapper assumes flat ground; observed support persists in this static course.
There is no loop closure, learned semantic model, dynamic obstacle benchmark,
rough-terrain controller, or real hardware validation. Camera quality is a
tracking heuristic, not a calibrated probability.

The old Isaac/Spot integration never demonstrated camera-driven A-to-B navigation.
Its historical logs are preserved in `archive/isaac-results/`.
