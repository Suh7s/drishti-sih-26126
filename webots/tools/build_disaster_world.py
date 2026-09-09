"""Procedural builder for Nepal Flood Disaster World in Webots.

Creates high-detail realistic textures and constructs a post-flood river valley
scene complete with:
- Alluvial flood channel, mud deposits, and gravel terraces
- Large river boulders, swept timber logs, and collapsed concrete ruins
- Undulating banks, slope contours, and muddy water hazard zones
- Overcast Himalayan post-storm lighting, ground mist, and distant mountain peaks
- DRISHTI rover with stereo cameras, non-supervisor controller
- Dedicated Supervisor evaluator recording ground truth and cinematic shots
"""
from pathlib import Path
import math
import json
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
TEXTURES_DIR = ROOT / "worlds" / "textures"
TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(26126)


def build_textures():
    print("Generating Nepal flood disaster textures...")
    size = 1024

    # 1. Mud / Silt Texture
    noise = np.zeros((size, size), np.float32)
    for n, strength in [(16, 15), (64, 20), (256, 25), (1024, 10)]:
        noise += cv2.resize(rng.normal(0, strength, (n, n)).astype(np.float32), (size, size))

    mud_base = np.clip(np.array([45, 62, 78])[None, None, :] + noise[:, :, None], 0, 255).astype(np.uint8)
    # Add water streaks and silt flow lines
    for _ in range(80):
        y = rng.integers(0, size)
        x = rng.integers(0, size)
        cv2.ellipse(mud_base, (int(x), int(y)), (int(rng.integers(20, 90)), int(rng.integers(3, 10))),
                    float(rng.integers(-20, 20)), 0, 360, (30, 42, 54), -1)
    cv2.imwrite(str(TEXTURES_DIR / "nepal_mud.jpg"), mud_base)

    # 2. Himalayan Boulder / Granite Texture
    boulder_tex = np.zeros((size, size, 3), dtype=np.uint8)
    b_noise = np.zeros((size, size), np.float32)
    for n, strength in [(8, 25), (32, 25), (128, 20), (512, 12)]:
        b_noise += cv2.resize(rng.normal(0, strength, (n, n)).astype(np.float32), (size, size))
    boulder_tex = np.clip(np.array([115, 120, 125])[None, None, :] + b_noise[:, :, None], 0, 255).astype(np.uint8)
    # Mineral veins / cracks
    for _ in range(25):
        pt1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        pt2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(boulder_tex, pt1, pt2, (80, 85, 90), thickness=int(rng.integers(1, 4)))
    cv2.imwrite(str(TEXTURES_DIR / "boulder.jpg"), boulder_tex)

    # 3. Composite Disaster Terrain Texture (4096 x 2048)
    W, H = 4096, 2048
    terrain_tex = np.zeros((H, W, 3), dtype=np.uint8)

    # Base gravel/soil bed
    gravel = cv2.resize(mud_base, (W, H))
    yy, xx = np.indices((H, W))

    # World coordinate mapping: X in [-7, 17], Y in [-9, 9]
    wx = (xx / W) * 24.0 - 7.0
    wy = (0.5 - yy / H) * 18.0

    # Flood gully path: riverbed snaking through the center
    gully_center = 0.6 * np.sin(wx * 0.45) - 0.2
    dist_to_gully = np.abs(wy - gully_center)

    # Mud blend in the alluvial channel (dark muddy brown)
    mud_mask = np.clip(1.0 - dist_to_gully / 2.8, 0.0, 1.0)[:, :, None]
    mud_color = np.array([35, 52, 68], dtype=np.float32)

    # Bank vegetation on the high outer ground
    veg_mask = np.clip((dist_to_gully - 3.2) / 2.5, 0.0, 1.0)[:, :, None]
    grass_color = np.array([48, 92, 55], dtype=np.float32)

    # Silt & rocky path along navigable corridor
    gravel_f = gravel.astype(np.float32)
    composite = gravel_f * (1.0 - mud_mask * 0.7) + mud_color * (mud_mask * 0.7)
    composite = composite * (1.0 - veg_mask) + (gravel_f * 0.3 + grass_color * 0.7) * veg_mask

    # Add scattered pebble details throughout
    pebbles = np.clip(composite, 0, 255).astype(np.uint8)
    for _ in range(4000):
        px = rng.integers(0, W)
        py = rng.integers(0, H)
        shade = int(rng.integers(70, 190))
        cv2.circle(pebbles, (int(px), int(py)), int(rng.integers(1, 4)), (shade, shade - 5, shade - 10), -1)

    cv2.imwrite(str(TEXTURES_DIR / "nepal_terrain.jpg"), pebbles)
    print("Textures generated successfully.")


def shape(geom, color, extra=""):
    return f"Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 0.88 metalness 0.05 {extra} }} geometry {geom} }}"


def pose(x, y, z, geom, color, extra="", rot=None):
    rot_str = f"rotation {rot} " if rot else ""
    return f"Pose {{ translation {x} {y} {z} {rot_str}children [ {shape(geom, color, extra)} ] }}"


def terrain_height(x, y):
    """Deterministic collidable terrain; flat surveyed launch zone."""
    x, y = np.asarray(x), np.asarray(y)
    ramp = .23 * np.exp(-((x - 5.2) / 1.8) ** 2)
    banks = .65 * (1 - np.exp(-(np.maximum(np.abs(y) - 2.5, 0) / 1.7) ** 2))
    roughness = .018 * np.sin(2.0 * x) * np.sin(1.8 * y)
    start_blend = np.clip((np.sqrt(x*x + y*y) - 1.6) / 1.4, 0, 1)
    return (ramp + banks + roughness) * start_blend


def build_world():
    build_textures()
    print("Assembling disaster.wbt scene...")

    w = ["""#VRML_SIM R2025a utf8
EXTERNPROTO "../protos/Oak.proto"
WorldInfo { basicTimeStep 32 randomSeed 26126 }
DEF SHOT Viewpoint { orientation -0.26 0.35 0.89 1.15 position -4.0 -5.5 3.8 fieldOfView 0.85 }
Background { skyColor [ 0.38 0.46 0.52 ] }
DirectionalLight { direction -0.35 0.55 -0.75 color 0.98 0.90 0.78 intensity 1.6 ambientIntensity 0.5 castShadows TRUE }
Fog { color 0.42 0.48 0.52 visibilityRange 48 }
"""]

    # Visible and collidable geometry use the SAME height field.
    xx, yy = np.meshgrid(np.linspace(-7, 17, 121), np.linspace(-9, 9, 91))
    heights = terrain_height(xx, yy)
    values = ' '.join(f'{z:.5f}' for z in heights.ravel())
    w.append(f"""Solid {{ translation -7 -9 0 name "ground_basin"
      children [ Shape {{ appearance PBRAppearance {{
        baseColorMap ImageTexture {{ url [ "textures/nepal_terrain.jpg" ] }}
        roughness .95 metalness 0
      }} geometry DEF BASIN_HEIGHTFIELD ElevationGrid {{
        xDimension 121 yDimension 91 xSpacing .2 ySpacing .2
        height [ {values} ] thickness 1
      }} }} ] boundingObject USE BASIN_HEIGHTFIELD }}""")
    manifest = {'schema': 1, 'rover_enclosing_radius_m': .42,
                'terrain': {'type': 'ElevationGrid', 'sample_spacing_m': .2,
                            'min_height_m': float(heights.min()), 'max_height_m': float(heights.max())},
                'hazards': [
                    {'name':'boulder', 'kind':'box', 'center':[3.5,.55], 'half_size':[.4,.4]},
                    {'name':'fallen_log', 'kind':'box', 'center':[7.2,1.3], 'half_size':[.24,.9]},
                    {'name':'ruin', 'kind':'box', 'center':[8.2,-1.4], 'half_size':[.375,.55], 'yaw':.42},
                    {'name':'water_exclusion', 'kind':'circle', 'center':[5.8,.85], 'radius':.85,
                     'scope':'visual no-go zone; no fluid or soil mechanics'}]}

    # Murky Flood Water Channel / Mud Hazard Zone
    # Low-lying mud puddle running along north flood terrace at x=5.8, y=0.85
    w.append(f"""DEF HAZARD_MUD_POOL Solid {{
  translation 5.8 0.85 {float(terrain_height(5.8,.85)) + .008}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 0.15 0.22 0.20
        roughness 0.12
        metalness 0.40
      }}
      geometry Cylinder {{ radius 0.85 height 0.005 subdivision 32 }}
    }}
  ]
  name "hazard_mud_water"
}}""")

    # Disaster Obstacle 1: Large fallen Himalayan boulder
    w.append(f"""DEF HAZARD_0 Solid {{
  translation 3.5 0.55 {float(terrain_height(3.5,.55)) + .325}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor .38 .40 .42
        roughness 0.90
        metalness 0.05
        baseColorMap ImageTexture {{ url [ "textures/boulder.jpg" ] }}
      }}
      geometry Box {{ size 0.80 0.80 0.65 }}
    }}
  ]
  name "hazard_boulder_0"
  boundingObject Box {{ size 0.80 0.80 0.65 }}
}}""")

    # Disaster Obstacle 2: Fallen swept timber tree trunk
    w.append(f"""DEF HAZARD_1 Solid {{
  translation 7.2 1.3 {float(terrain_height(7.2,1.3)) + .24}
  rotation 1 0 0 1.57079632679
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor .28 .22 .16
        roughness 0.85
        metalness 0.02
      }}
      geometry Cylinder {{ radius 0.24 height 1.8 subdivision 24 }}
    }}
  ]
  name "hazard_log_1"
  boundingObject Cylinder {{ radius 0.24 height 1.8 subdivision 24 }}
}}""")

    # Disaster Obstacle 3: Collapsed concrete building ruin
    w.append(f"""DEF HAZARD_2 Solid {{
  translation 8.2 -1.4 {float(terrain_height(8.2,-1.4)) + .32}
  rotation 0 0 1 0.42
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor .48 .46 .44
        roughness 0.92
        metalness 0.05
      }}
      geometry Box {{ size 0.75 1.1 0.64 }}
    }}
  ]
  name "hazard_ruin_2"
  boundingObject Box {{ size 0.75 1.1 0.64 }}
}}""")

    # Additional small rocks & river debris scattered naturally
    small_rocks = [
        (2.4, 1.8, 0.15, 0.35, 0.40, 0.30),
        (5.4, -2.1, 0.18, 0.45, 0.38, 0.35),
        (8.2, 2.1, 0.14, 0.38, 0.42, 0.28),
        (9.4, -0.7, 0.16, 0.40, 0.35, 0.32),
    ]
    for i, (rx, ry, rz, sx, sy, sz) in enumerate(small_rocks):
        rz = float(terrain_height(rx,ry)) + sz / 2
        manifest['hazards'].append({'name':f'debris_{i}', 'kind':'box', 'center':[rx,ry], 'half_size':[sx/2,sy/2]})
        w.append(f"""Solid {{
  translation {rx} {ry} {rz}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor .35 .37 .39
        roughness 0.9
        baseColorMap ImageTexture {{ url [ "textures/boulder.jpg" ] }}
      }}
      geometry Box {{ size {sx} {sy} {sz} }}
    }}
  ]
  name "debris_rock_{i}"
  boundingObject Box {{ size {sx} {sy} {sz} }}
}}""")

    # Start pad (clean launch zone at x=0, y=0)
    w.append(pose(0, 0, 0.005, "Cylinder { radius 0.65 height 0.01 subdivision 48 }", ".18 .65 .55"))

    # Goal Disaster Relief Checkpoint Marker (high-visibility safety orange beacon)
    # Pole and flag marker are offset to the edge of the landing circle
    w.append(pose(10.0, 0, 0.008, "Cylinder { radius 0.75 height 0.016 subdivision 48 }", ".95 .45 .12"))
    w.append(pose(10.55, 0.45, 0.45, "Cylinder { radius 0.035 height 0.90 subdivision 16 }", ".15 .18 .20"))
    w.append(pose(10.55, 0.45, 0.92, "Box { size 0.42 0.02 0.24 }", ".98 .42 .10"))

    # DRISHTI ROVER
    w.append("""DEF ROVER Robot {
  translation 0 0 0.13
  name "DRISHTI rover"
  controller "rover"
  supervisor FALSE
  children [
    Pose { translation 0 0 0.1 children [ Shape { appearance PBRAppearance { baseColor .09 .13 .15 roughness 0.85 metalness 0 } geometry Box { size .52 .34 .17 } } ] }
    Pose { translation 0 0 0.2 children [ Shape { appearance PBRAppearance { baseColor .91 .52 .12 roughness 0.85 metalness 0 } geometry Box { size .38 .32 .05 } } ] }
    Pose { translation 0.2 0 0.38 children [ Shape { appearance PBRAppearance { baseColor .1 .14 .16 roughness 0.85 metalness 0 } geometry Box { size .035 .04 .43 } } ] }
    Pose { translation 0.25 0 0.62 children [ Shape { appearance PBRAppearance { baseColor .06 .08 .10 roughness 0.85 metalness 0 } geometry Box { size .06 .28 .055 } } ] }
    Pose { translation 0.267 -0.1 0.15 children [ Shape { appearance PBRAppearance { baseColor 1 .9 .63 roughness 0.85 metalness 0 } geometry Sphere { radius .027 subdivision 2 } } ] }
    Pose { translation 0.267 0.1 0.15 children [ Shape { appearance PBRAppearance { baseColor 1 .9 .63 roughness 0.85 metalness 0 } geometry Sphere { radius .027 subdivision 2 } } ] }
    Pose { translation -0.025 -0.175 0.13 children [ Shape { appearance PBRAppearance { baseColor .12 .7 .64 roughness 0.85 metalness 0 } geometry Box { size .32 .008 .035 } } ] }
    Pose { translation -0.025 0.175 0.13 children [ Shape { appearance PBRAppearance { baseColor .12 .7 .64 roughness 0.85 metalness 0 } geometry Box { size .32 .008 .035 } } ] }
    Pose { translation -0.12 0 0.227 children [ Shape { appearance PBRAppearance { baseColor .14 .18 .19 roughness 0.85 metalness 0 } geometry Box { size .009 .23 .005 } } ] }
    Pose { translation -0.06 0 0.227 children [ Shape { appearance PBRAppearance { baseColor .14 .18 .19 roughness 0.85 metalness 0 } geometry Box { size .009 .23 .005 } } ] }
    Pose { translation 0 0 0.227 children [ Shape { appearance PBRAppearance { baseColor .14 .18 .19 roughness 0.85 metalness 0 } geometry Box { size .009 .23 .005 } } ] }
    Pose { translation 0.06 0 0.227 children [ Shape { appearance PBRAppearance { baseColor .14 .18 .19 roughness 0.85 metalness 0 } geometry Box { size .009 .23 .005 } } ] }
    Pose { translation 0.12 0 0.227 children [ Shape { appearance PBRAppearance { baseColor .14 .18 .19 roughness 0.85 metalness 0 } geometry Box { size .009 .23 .005 } } ] }
    Camera { translation .29 0.08 .62 rotation 0 1 0 0.78539816339 name "left" width 640 height 400 fieldOfView 1.4 near .03 }
    Camera { translation .29 -0.08 .62 rotation 0 1 0 0.78539816339 name "right" width 640 height 400 fieldOfView 1.4 near .03 }
    HingeJoint { jointParameters HingeJointParameters { axis 0 1 0 anchor 0.18 0.23 0 }
      device [ RotationalMotor { name "front_left" maxVelocity 15 maxTorque 30 } ]
      endPoint Solid { translation 0.18 0.23 0 rotation 1 0 0 1.57079632679
        children [ Shape { appearance PBRAppearance { baseColor .035 .045 .05 roughness 0.85 metalness 0 } geometry Cylinder { radius .13 height .10 subdivision 32 } }
        Pose { translation 0 0 0.052 children [ Shape { appearance PBRAppearance { baseColor .48 .53 .55 roughness 0.85 metalness 0 } geometry Cylinder { radius .071 height .006 subdivision 12 } } ] } ]
        name "front_left_wheel" boundingObject Cylinder { radius .13 height .1 subdivision 32 }
        physics Physics { density -1 mass .7 } } }
    HingeJoint { jointParameters HingeJointParameters { axis 0 1 0 anchor -0.18 0.23 0 }
      device [ RotationalMotor { name "rear_left" maxVelocity 15 maxTorque 30 } ]
      endPoint Solid { translation -0.18 0.23 0 rotation 1 0 0 1.57079632679
        children [ Shape { appearance PBRAppearance { baseColor .035 .045 .05 roughness 0.85 metalness 0 } geometry Cylinder { radius .13 height .10 subdivision 32 } }
        Pose { translation 0 0 0.052 children [ Shape { appearance PBRAppearance { baseColor .48 .53 .55 roughness 0.85 metalness 0 } geometry Cylinder { radius .071 height .006 subdivision 12 } } ] } ]
        name "rear_left_wheel" boundingObject Cylinder { radius .13 height .1 subdivision 32 }
        physics Physics { density -1 mass .7 } } }
    HingeJoint { jointParameters HingeJointParameters { axis 0 1 0 anchor 0.18 -0.23 0 }
      device [ RotationalMotor { name "front_right" maxVelocity 15 maxTorque 30 } ]
      endPoint Solid { translation 0.18 -0.23 0 rotation 1 0 0 1.57079632679
        children [ Shape { appearance PBRAppearance { baseColor .035 .045 .05 roughness 0.85 metalness 0 } geometry Cylinder { radius .13 height .10 subdivision 32 } }
        Pose { translation 0 0 0.052 children [ Shape { appearance PBRAppearance { baseColor .48 .53 .55 roughness 0.85 metalness 0 } geometry Cylinder { radius .071 height .006 subdivision 12 } } ] } ]
        name "front_right_wheel" boundingObject Cylinder { radius .13 height .1 subdivision 32 }
        physics Physics { density -1 mass .7 } } }
    HingeJoint { jointParameters HingeJointParameters { axis 0 1 0 anchor -0.18 -0.23 0 }
      device [ RotationalMotor { name "rear_right" maxVelocity 15 maxTorque 30 } ]
      endPoint Solid { translation -0.18 -0.23 0 rotation 1 0 0 1.57079632679
        children [ Shape { appearance PBRAppearance { baseColor .035 .045 .05 roughness 0.85 metalness 0 } geometry Cylinder { radius .13 height .10 subdivision 32 } }
        Pose { translation 0 0 0.052 children [ Shape { appearance PBRAppearance { baseColor .48 .53 .55 roughness 0.85 metalness 0 } geometry Cylinder { radius .071 height .006 subdivision 12 } } ] } ]
        name "rear_right_wheel" boundingObject Cylinder { radius .13 height .1 subdivision 32 }
        physics Physics { density -1 mass .7 } } }
  ]
  boundingObject Pose { translation 0 0 .10 children [ Box { size .52 .34 .17 } ] }
  physics Physics { density -1 mass 6 centerOfMass [ 0 0 .06 ] }
}""")

    # Surrounding mountain vegetation: Oak trees placed along the high ridges
    for i in range(40):
        tx = float(rng.uniform(-4.5, 17.5))
        ty = float(rng.choice([-1, 1]) * rng.uniform(3.4, 8.8))
        yaw = float(rng.uniform(0, 6.28))
        manifest['hazards'].append({'name':f'tree_{i}', 'kind':'circle', 'center':[round(tx,3),round(ty,3)], 'radius':.5})
        w.append(f'Oak {{ translation {tx:.3f} {ty:.3f} {float(terrain_height(tx,ty)):.3f} rotation 0 0 1 {yaw:.3f} name "nepal_tree_{i}" }}')

    # Himalayan Mountain Peak Silhouettes in the distance
    mountain_positions = [
        (-10, 22, 6, 14, 16),
        (0, 24, 7, 16, 18),
        (10, 25, 8, 18, 20),
        (20, 23, 6, 15, 17),
        (-6, -22, 6, 14, 15),
        (6, -24, 7, 16, 17),
        (16, -23, 6, 15, 16),
    ]
    for i, (mx, my, mz, rad, height) in enumerate(mountain_positions):
        w.append(pose(mx, my, mz, f"Cone {{ bottomRadius {rad} height {height} subdivision 8 }}", ".26 .34 .36"))

    # Dedicated Evaluation Supervisor Controller
    w.append('Robot { name "evaluation_only" supervisor TRUE controller "evaluator" }')

    out_file = ROOT / "worlds" / "disaster.wbt"
    out_file.write_text("\n".join(w))
    out_file.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Created disaster world at: {out_file}")
    return out_file


if __name__ == "__main__":
    build_world()
