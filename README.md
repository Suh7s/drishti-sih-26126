# DRISHTI

**Camera-driven autonomous navigation, running on a Mac in Webots.**

SIH problem 26126 · Bharat Electronics Limited · GPS-denied outdoor UGV navigation.

The current demonstrator is a four-wheel rover with rectified stereo cameras,
metric visual odometry, observed-ground mapping and collision-aware A* control.
The robot controller receives RGB images only. A separate supervisor records
simulator truth for evaluation.

## Measured physics run

The saved flat, static two-obstacle run reached its destination in **59.33 s**,
with **8.88 mm position RMSE** and **10.09 cm conservative obstacle clearance**.
These are results from one controlled simulation, not real-world performance.

See [evaluation and raw logs](results/validated_webots/) and
[the run image](results/validated_webots/scene.jpg).

## Run on macOS or Linux

Install [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a), then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py test
python webots/launch.py
```

Pass `--webots /path/to/Webots.app` if Webots is installed elsewhere.

## Scope

Validated: flat static course, known initial pose and checked launch pad, classical
stereo and visual odometry. Not yet demonstrated: learned semantic perception,
loop-closing SLAM, rough terrain, dynamic obstacles or physical hardware.

The `isaac/` adapter and older `results/isaac_*` folders are experimental history.
They do not establish successful Spot navigation. The existing `demo/` viewer is
an older synthetic planning replay; it is not footage of the Webots run.
