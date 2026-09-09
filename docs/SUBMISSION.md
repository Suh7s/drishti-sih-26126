# Submission runbook

1. Install dependencies and Webots R2025a; run `python run.py test`.
2. For the recorded, verified baseline, run `python run.py demo`. Keep the local browser open before presenting. The video also plays directly from `demo/mission.mp4`.
3. For the new terrain experiment, run `python webots/launch.py --world disaster --record --exit`. Allow time for stereo processing and scene rendering; simulation time differs from wall-clock time.
4. Inspect that run's `evaluation.json`. Present success only when its `success` field is true. Keep its `config.json`, `scene_manifest.json`, navigation log and independent truth log together.
5. Explain the architecture: cameras feed navigation; only the independent evaluator sees simulator truth. Calibration and a clear launch footprint are known priors.
6. State the tested scene and scope. Do not describe the baseline movie as a disaster-terrain run, the classifier's procedural accuracy as real-world accuracy, or the rover as field-ready.

For a technical jury, lead with the recorded result, then show the sensor-to-command pipeline, one safety regression and the independent error plot. Describe remaining experiments as specific research questions rather than unsupported performance promises.
