# DRISHTI: Ubuntu handoff

Target: Ubuntu, NVIDIA RTX 5080, 32 GB RAM, Isaac Sim 6.0.1.
SIH problem 26126, Bharat Electronics Limited. Deadline: tomorrow.
Team will prepare its own presentation. No physical robot is available.

## Start the next Codex CLI session

Extract this package on Ubuntu. Open Codex CLI in the extracted `drishti` directory.
Paste the contents of `UBUNTU_HANDOFF.md` as the next task. Let that session inspect
the installed Isaac Sim code and fix/run the integration on the actual RTX machine.

The camera pipeline and planning harness are implemented. The Isaac adapter and
cinematic camera have NOT been run in Isaac Sim. They are starting code, not a
verified complete navigation system. Do not start by recording a submission video.

## Flow on Ubuntu

1. Inspect `docs/STATUS.md`, then locate the existing Isaac Sim 6.0.1 installation.
2. Run NVIDIA's installed Spot standalone example unchanged. Confirm stable standing
   and walking before modifying the scene or controller.
3. Run portable tests with an environment containing NumPy and OpenCV.
4. Inspect and reconcile `isaac/run_spot.py` with the installed 6.0.1 APIs.
5. Run capture mode with stationary Spot. Verify stereo synchronization, sign of
   disparity, camera mounting, metric depth and stable estimated ground height.
6. Fix the known mapping/coverage problems listed in STATUS before movement.
7. Begin with an obstacle-free two-metre route. Add one obstacle only after the
   camera-based pose and stopping supervisor are stable.
8. Run the same route repeatedly, inspect failures and record telemetry.
9. Enable the independent cinematic recorder after the navigation run works.
10. Preserve raw evidence and label every video by its actual mode and inputs.

## Commands after locating the installation

Replace the example path with the real folder containing `python.sh`.

```bash
export ISAAC_SIM_PATH=/absolute/path/to/isaac-sim
"$ISAAC_SIM_PATH/python.sh" -c 'import numpy, cv2; print(numpy.__version__, cv2.__version__)'
"$ISAAC_SIM_PATH/python.sh" run.py test
bash isaac/launch_ubuntu.sh --mode capture --duration 15
bash isaac/launch_ubuntu.sh --mode vision --goal-x 2 --duration 45
bash isaac/launch_ubuntu.sh --mode vision --goal-x 10 --duration 90 --cinematic
```

The launch commands are intended entry points. Runtime fixes are expected before
they succeed. If OpenCV is missing, the CLI should first inspect Isaac's bundled
Python and NumPy constraints before adding a compatible OpenCV package. Do not
upgrade or replace Isaac's bundled Torch/CUDA/NumPy stack blindly.

Each run defaults to `results/isaac_run`. Use a new `--output` directory for each
attempt to preserve telemetry and avoid overwriting recordings.

## Delivered files

- `src/drishti/`: navigation, stereo, odometry, mapping and integration code.
- `isaac/run_spot.py`: experimental 6.0.1 scene and Spot adapter.
- `isaac/cinematic.py`: independent tracking camera and MP4 recorder.
- `tests/`: portable algorithm and synthetic-image integration tests.
- `demo/index.html`: offline replay of the 2-D planning experiment.
- `results/metrics.csv`: 24 planning executions, with explicit synthetic provenance.
- `results/replays.json`: recorded planning trajectories, not Isaac recordings.

Open `demo/index.html` directly for the optional offline replay. It is only a
planning demonstration and must not be presented as the Isaac Sim result.
