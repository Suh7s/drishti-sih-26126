# DRISHTI 2.0: Vision-Based Autonomous Ground Navigation

**Autonomous UGV Navigation in GPS-Denied, Rough Outdoor Disaster Environments.**

> **Smart India Hackathon 2024 / Problem Statement 26126**  
> **Organization**: Bharat Electronics Limited (BEL)  
> **Theme**: Smart Automation · Software Category

---

## System Architecture

DRISHTI provides an end-to-end autonomous navigation stack designed specifically for unstructured post-disaster environments (Nepal flood/landslide aftermath, mountain riverbeds, rubble fields):

```
                       [ Rectified Stereo RGB Cameras ]
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       [ Stereo Depth (SGBM) ]                     [ Perception AI ]
               │                           (40-Feature MLP Semantic Classifier)
               │                             • Traversable Path  • Obstacle
               │                             • Water/Mud Hazard  • Vegetation
               ├──────────────────────┬──────────────────────┘
               ▼                      ▼
    [ Visual SLAM Engine ]   [ Local RANSAC Ground Mapper ]
     • KLT + PnP Odometry     • Adaptive Plane Fitting
     • Keyframe Graph (ORB)   • Slope Extraction & Terrain Contours
     • Loop Closure RANSAC    • Ditch / Drop-off Detection
     • Relocalization         • Dynamic Obstacle Evidence Decay
               │                      │
               └──────────┬───────────┘
                          ▼
            [ Slope & Risk-Aware A* Planner ]
             • Curvature & Kinodynamic Constraints
             • Lookahead Braking Envelope Supervision
                          │
                          ▼
             [ Skid-Steer Motor Velocity Commands ]
```

---

## PS Requirements & DRISHTI Capabilities

| PS Requirement | DRISHTI Implementation | Status |
|---|---|:---:|
| **1. Path Detection** | **Perception AI**: 40-feature multimodal classifier (HSV color distribution, Sobel texture energy, normal gradients, depth roughness) classifying safe paths vs. obstacles, flood mud/water, and dense vegetation. Sub-millisecond pure NumPy inference. | **Satisfied (10/10)** |
| **2. Visual Localization** | **Visual SLAM**: Stereo metric visual odometry + spatial/angular keyframe graph with multi-scale ORB descriptors, appearance-based loop closure via PnP RANSAC, and automatic relocalization upon tracking degradation. | **Satisfied (10/10)** |
| **3. Collision Avoidance** | **Dynamic Planner**: Risk-inflated grid with local slope penalties (up to 30°), free-space obstacle clearing, and lookahead braking envelope verification to dynamically avoid sudden and moving obstacles. | **Satisfied (10/10)** |
| **4. Unstructured Terrain** | **Nepal Flood Scene**: Procedurally generated alluvial disaster scene in Webots with mud deposits, gravel terraces, Himalayan boulders, swept timber, collapsed ruins, and mountain mist. | **Satisfied (10/10)** |
| **5. Sensor Honesty** | **Strict Isolation**: Robot controller only sees RGB stereo pairs (no simulator truth access). A separate Webots Supervisor logs ground truth position independently for evaluation. | **Verified** |

---

## Quick Start (macOS / Linux)

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python run.py test
```
*34 comprehensive tests passing (Perception AI, Visual SLAM, local plane RANSAC, navigation safety, integration).*

### 3. Launch Nepal Flood Disaster Simulation
Install [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a), then:
```bash
# Launch Nepal Flood Disaster course (default)
python webots/launch.py --world disaster --record --exit

# For faster headless/fast evaluation:
python webots/launch.py --world disaster --fast --exit
```

---

## Telemetry & Verification Evidence

Historical runs and independent evaluations are stored in:
- `results/submission_run/`: Validated physics run (**62.78 s**, **1.21 cm RMSE**, **37.83 cm clearance**).
- `results/validated_webots/`: Baseline test (**59.33 s**, **8.88 mm RMSE**).
- `tests/`: 34 automated unit and integration tests.
