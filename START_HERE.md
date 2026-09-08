# Start here

1. Install Webots R2025a from its official distribution.
2. Create a Python environment and install `requirements.txt`.
3. Run `python run.py test`.
4. Launch `python webots/launch.py --record --exit`.

Each launch creates a separate `results/webots_run_...` folder with navigation
logs, independent simulator positions, stereo diagnostics and an evaluation.
`--record` also records the main camera and stereo perception. `--exit` closes
Webots once the run and recording finish.

Use `--webots /path/to/Webots.app` for a nonstandard installation, or
`--world benchmark` for the original simpler course. Use `--fast` to let the
simulator run as quickly as the computer allows; metrics still use simulation time.

The mission is a static, flat 8-metre goal displacement, with two obstacles.
Camera-based estimates control the rover. Simulator truth only scores the run
and positions the separate cinematic camera.
