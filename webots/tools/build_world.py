"""Deterministic, self-contained Webots course. No downloaded PROTO assets."""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(26126)
size = 1024
noise = np.zeros((size, size), np.float32)
for n, strength in [(16, 12), (64, 16), (256, 20), (1024, 8)]:
    noise += cv2.resize(rng.normal(0, strength, (n, n)).astype(np.float32), (size, size))
soil = np.clip(np.array([100, 111, 122])[None, None, :] + noise[:, :, None], 0, 255).astype(np.uint8)
for _ in range(3500):
    x, y = rng.integers(0, size, 2)
    shade = int(rng.integers(45, 180))
    cv2.ellipse(soil, (int(x), int(y)), (int(rng.integers(1, 5)), int(rng.integers(1, 3))),
                float(rng.integers(180)), 0, 360, (shade, shade+5, shade+10), -1)
cv2.imwrite(str(ROOT/'worlds/textures/soil.jpg'), soil)

def shape(geometry, color, extra=''):
    return f'Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 0.85 metalness 0 {extra} }} geometry {geometry} }}'

def pose(x, y, z, geometry, color):
    return f'Pose {{ translation {x} {y} {z} children [ {shape(geometry, color)} ] }}'

w = ['''#VRML_SIM R2025a utf8
WorldInfo { basicTimeStep 32 randomSeed 26126 }
DEF SHOT Viewpoint { orientation -0.310 0.175 0.934 2.115 position -3 -5 3.6 fieldOfView 0.85 }
Background { skyColor [ 0.57 0.70 0.78 ] }
DirectionalLight { direction -0.4 0.5 -0.85 color 1 0.91 0.78 intensity 1.5 ambientIntensity 0.55 castShadows TRUE }
Fog { color 0.57 0.70 0.78 visibilityRange 65 }
Solid { translation 5 0 -0.05 children [
Shape { appearance PBRAppearance { baseColorMap ImageTexture { url [ "textures/soil.jpg" ] } textureTransform TextureTransform { scale 12 9 } roughness 1 metalness 0 } geometry Box { size 24 18 0.1 } }
] name "ground" boundingObject Box { size 24 18 0.1 } }
DEF ROVER Robot {
 translation 0 0 0.13
 name "DRISHTI rover"
 controller "rover"
 supervisor FALSE
 children [
''']
w += [pose(0,0,.10,'Box { size .52 .34 .17 }','.09 .13 .15'),
      pose(0,0,.20,'Box { size .38 .32 .05 }','.91 .52 .12'),
      pose(.20,0,.38,'Box { size .035 .04 .43 }','.1 .14 .16'),
      pose(.25,0,.62,'Box { size .06 .28 .055 }','.06 .08 .10')]
for side, y in [('left', .08), ('right', -.08)]:
    w.append(f'Camera {{ translation .29 {y} .62 rotation 0 1 0 0.78539816339 name "{side}" width 640 height 400 fieldOfView 1.4 near .03 }}')
for name,x,y in [('front_left',.18,.23),('rear_left',-.18,.23),('front_right',.18,-.23),('rear_right',-.18,-.23)]:
    w.append(f'''HingeJoint {{ jointParameters HingeJointParameters {{ axis 0 1 0 anchor {x} {y} 0 }}
    device [ RotationalMotor {{ name "{name}" maxVelocity 15 maxTorque 12 }} ]
    endPoint Solid {{ translation {x} {y} 0 rotation 1 0 0 1.57079632679
    children [ {shape('Cylinder { radius .13 height .10 subdivision 32 }','.035 .045 .05')}
    {pose(0,0,.052,'Cylinder { radius .071 height .006 subdivision 12 }','.48 .53 .55')} ]
    name "{name}_wheel" boundingObject Cylinder {{ radius .13 height .1 subdivision 32 }}
    physics Physics {{ density -1 mass .7 }} }} }}''')
w.append('] boundingObject Pose { translation 0 0 .10 children [ Box { size .52 .34 .17 } ] } physics Physics { density -1 mass 6 centerOfMass [ 0 0 .06 ] } }')
# First course is deliberately flat: isolate vision and control before terrain claims.
for i,(x,y,sx,sy,h) in enumerate([(3.5,0,.8,.8,.65),(6.4,1.2,.7,1.0,.8)]):
    w.append(f'DEF HAZARD_{i} Solid {{ translation {x} {y} {h/2} children [ {shape(f"Box {{ size {sx} {sy} {h} }}", ".31 .34 .32", "baseColorMap ImageTexture { url [ \"textures/soil.jpg\" ] }")} ] name "rock_{i}" boundingObject Box {{ size {sx} {sy} {h} }} }}')
for i in range(38):
    x=float(rng.uniform(-3,15)); y=float(rng.choice([-1,1])*rng.uniform(3.4,7.8)); h=float(rng.uniform(2.2,4.8))
    w.append(pose(x,y,h*.25,f'Cylinder {{ radius .11 height {h*.5} }}','.23 .18 .12'))
    for z,r in [(h*.50,h*.25),(h*.7,h*.20),(h*.87,h*.13)]:
        w.append(pose(x,y,z,f'Cone {{ bottomRadius {r} height {h*.48} subdivision 10 }}','.16 .27 .20'))
for x in [0,8]:
    w.append(pose(x,0,.004,'Cylinder { radius .48 height .006 subdivision 48 }','.12 .65 .59'))
w.append('Robot { name "evaluation_only" supervisor TRUE controller "evaluator" }')
(ROOT/'worlds/drishti.wbt').write_text('\n'.join(w))
print(ROOT/'worlds/drishti.wbt')
