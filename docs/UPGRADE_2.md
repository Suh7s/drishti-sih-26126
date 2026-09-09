# DRISHTI 2.0 engineering review

The integration starts at upstream commit `07afb3c`, preserving the additional Desktop edits listed in [IMPORTED_DESKTOP_CHANGES.md](IMPORTED_DESKTOP_CHANGES.md). The original Desktop checkout remains untouched.

## Changes with practical consequences

1. **Terrain affects the wheels.** The visible surface and collision surface share one Webots ElevationGrid, sampled every 20 cm. The course has a gentle 23 cm central crest and higher outer banks; a flat surveyed launch region remains explicit. Decorative tilted boxes no longer stand in for traversed terrain. The fallen log is horizontal and debris rests at the generated ground height.
2. **Evaluation matches the scene.** The builder exports the same obstacle locations, sizes and rotations to an independent manifest. Scoring includes all debris, tree exclusion footprints and the water no-go region. Actual body tilt and elevation are measured from simulator truth, separately from the estimated local ground-plane slope. Conservative enclosing-circle clearance is a geometric metric, not a contact sensor.
3. **Training starts with images.** Procedural pinhole depth planes, material colors, textures, lighting variation and disparity dropout produce RGB/depth patches. Feature extraction is identical during training and runtime. A held-out scene seed is used for the 2,000-sample evaluation. This is a limited synthetic material domain; there is no claim of accurate real flood-water classification.
4. **Rotations remain rotations.** Loop corrections use SE(3) transforms with interpolated rotation vectors; elementwise rotation-matrix blending was removed. Keyframe landmarks move consistently, and occupancy is rebuilt after pose corrections. Loop candidates require a traveled revisit and checks are rate-limited. This remains a pose-chain approximation, not full graph optimization.
5. **Unknown means unknown.** Expired occupied cells lose observed-free status. Steep cells receive the same footprint inflation as obstacles. A* cannot pass through its own inflation margin or secretly substitute a different destination. Clearing occupancy requires three consecutive supporting frames.
6. **Ground follows elevation.** RANSAC is refit on inliers. Its height prior follows the visually estimated vehicle height. No plausible plane means no new ground support; the mapper does not silently revert to a global flat plane.
7. **Launches preserve provenance.** Each disaster launch backs up the current world, regenerates the canonical scene, records settings and copies the evaluation manifest. The evaluator rejects an unexpected start position. Model weights and their report are included in Python package data.

## Deliberate departures from the proposed plan

- The retained 40-feature representation combines HSV statistics/histograms, Sobel/Laplacian energy, color ratios, depth statistics and eight Gabor responses. It is not the proposed 24-bin-histogram layout.
- The training renderer produces material patches, not labeled full Webots scenes. Full-scene semantic accuracy needs pixel-level ground truth; obstacle positions alone cannot supply it.
- The water region is explicitly scored as an exclusion zone. It has no invisible collision wall and no simulated flood currents or sinking soil.
- A declared flat, clear launch footprint remains necessary because the downward-forward cameras do not observe directly beneath the robot. It is checked against stereo before movement.
- The rover retains historical ground support in this static scene. Temporal obstacle handling is implemented but a moving-obstacle benchmark is still needed.
- VisualSLAM returns a fourth diagnostics item; the pipeline adapts it explicitly. It is not falsely advertised as tuple-identical to VisualOdometry.

ElevationGrid follows the [Webots reference](https://cyberbotics.com/doc/reference/elevationgrid?version=released). Heights lie along local Z; visible and collision geometry share the same height samples.
