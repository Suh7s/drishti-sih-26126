"""Experimental Isaac Sim 6.0.1 adapter. Requires runtime verification on RTX.

Run with Isaac Sim's python.sh, not the system Python. Uses installed NVIDIA
Spot locomotion policy, rendered RGB images, SGBM depth and PnP visual odometry.
No scene-truth pose/depth enters navigation. --mode capture keeps zero commands.
No downloaded policy or Spot asset is redistributed in this package.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
parser=argparse.ArgumentParser()
parser.add_argument("--mode",choices=["capture","vision","policy_demo"],default="capture")
parser.add_argument("--headless",action="store_true")
parser.add_argument("--cinematic",action="store_true",help="Record a presentation camera separately from robot vision")
parser.add_argument("--duration",type=float,default=90)
parser.add_argument("--goal-x",type=float,default=10)
parser.add_argument("--goal-y",type=float,default=0)
parser.add_argument("--body-prim",default=None)
parser.add_argument("--output",default=str(ROOT/"results"/"isaac_run"))
args,kit_args=parser.parse_known_args()
from isaacsim import SimulationApp
app=SimulationApp({"headless":args.headless,"width":1280,"height":720})
output=Path(args.output);output.mkdir(parents=True,exist_ok=True)

try:
    print("DRISHTI phase: imports",flush=True)
    import numpy as np
    import cv2
    import omni.usd
    import omni.timeline
    from pxr import UsdGeom,UsdLux,UsdPhysics,Gf
    from isaacsim.core.experimental.materials import OmniPbrMaterial
    from isaacsim.core.experimental.prims import GeomPrim
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.robot.policy.examples")
    enable_extension("isaacsim.sensors.camera")
    app.update()
    from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    from drishti.vision import Calibration,camera_ros_mount
    from drishti.pipeline import CameraNavigation

    def rgb_u8(image):
        rgb=np.asarray(image[:,:,:3])
        if np.issubdtype(rgb.dtype,np.floating):
            rgb=np.clip(rgb*255.0,0,255)
        return rgb.astype(np.uint8)

    print("DRISHTI phase: extensions ready",flush=True)
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.0)

    def cube(path,position,size,color,collision=True):
        obj=UsdGeom.Cube.Define(stage,path);obj.CreateSizeAttr(1.0)
        obj.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)))
        obj.AddScaleOp().Set(Gf.Vec3f(*map(float,size)))
        obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        return obj

    cube("/World/Ground",(6,0,-.12),(32,24,.24),(.18,.23,.13))
    ground_material=OmniPbrMaterial("/World/Materials/ground")
    ground_material.set_input_values("diffuse_color_constant",[.75,.85,.55])
    GeomPrim("/World/Ground").apply_visual_materials(ground_material)
    dome=UsdLux.DomeLight.Define(stage,"/World/Sky");dome.CreateIntensityAttr(350)
    sun=UsdLux.DistantLight.Define(stage,"/World/Sun");sun.CreateIntensityAttr(1200)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(25,-35,-25))
    sun.CreateColorAttr(Gf.Vec3f(1.0,.88,.70))
    dome.CreateColorAttr(Gf.Vec3f(.76,.84,1.0))
    # Broad area lighting keeps the camera-only launch course textured without
    # changing collision geometry or supplying any navigation truth.
    key=UsdLux.SphereLight.Define(stage,"/World/Key");key.CreateIntensityAttr(900);key.CreateRadiusAttr(3.0)
    key.AddTranslateOp().Set(Gf.Vec3d(2.0,-2.5,5.5));key.CreateColorAttr(Gf.Vec3f(1.0,.78,.56))
    fill=UsdLux.SphereLight.Define(stage,"/World/Fill");fill.CreateIntensityAttr(450);fill.CreateRadiusAttr(3.0)
    fill.AddTranslateOp().Set(Gf.Vec3d(1.0,3.0,4.0));fill.CreateColorAttr(Gf.Vec3f(.62,.75,1.0))
    # Visual surface variation supplies image texture without hidden navigation labels.
    rng=np.random.default_rng(26126)
    for i in range(450):
        x,y=rng.uniform(-3,17),rng.uniform(-7,7)
        s=rng.uniform(.035,.16)
        color=tuple(rng.uniform(.16,.36,3))
        cube(f"/World/Texture/t{i}",(x,y,.001),(s,s,.001),color,False)
    # Meter-scale, non-colliding tonal patches make real stereo and PnP
    # repeatable over the verified flat launch route. They are visual only.
    for ix,x in enumerate(np.arange(-1.0,4.6,.28)):
        for iy,y in enumerate(np.arange(-2.1,2.2,.28)):
            tone=.55+.35*((ix*17+iy*29)%7)/6
            color=(tone*.78,tone*.98,tone*.58)
            path=f"/World/LaunchTexture/p{ix}_{iy}"
            cube(path,(float(x),float(y),.003),(.245,.245,.004),color,False)
            material=OmniPbrMaterial(f"/World/Materials/launch_{ix}_{iy}")
            material.set_input_values("diffuse_color_constant",[float(c) for c in color])
            GeomPrim(path).apply_visual_materials(material)
    for i,(x,y,s) in enumerate([(4.5,.1,.8),(7.5,-1.4,.9),(9.0,2.4,.7)]):
        cube(f"/World/Rocks/r{i}",(x,y,s*.4),(s,s*.8,s*.8),(.37,.36,.32))
    for i,x in enumerate(np.linspace(-1,15,12)):
        for side,y in enumerate((-4.8,4.8)):
            trunk=UsdGeom.Cylinder.Define(stage,f"/World/Trees/t{i}_{side}")
            trunk.CreateRadiusAttr(.17);trunk.CreateHeightAttr(3.0)
            trunk.AddTranslateOp().Set(Gf.Vec3d(float(x),y,1.5))
            trunk.CreateDisplayColorAttr([Gf.Vec3f(.23,.15,.08)])
            UsdPhysics.CollisionAPI.Apply(trunk.GetPrim())
            crown=UsdGeom.Sphere.Define(stage,f"/World/Trees/c{i}_{side}")
            crown.CreateRadiusAttr(1.15);crown.AddTranslateOp().Set(Gf.Vec3d(float(x),y,3.0))
            crown.CreateDisplayColorAttr([Gf.Vec3f(.12,.26,.10)])
    cube("/World/GoalMarker",(args.goal_x,args.goal_y,.008),(.75,.75,.015),(.15,.65,.65),False)

    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.deprecation_manager import import_module
    torch=import_module("torch")
    UsdPhysics.Scene.Define(stage,"/World/PhysicsScene")
    RenderingManager.set_dt(.05)
    # Match the installed standalone Spot example: policy inference and the
    # PhysX articulation run on the RTX device.
    SimulationManager.set_physics_sim_device("cuda")
    SimulationManager.set_physics_dt(.005)
    print("DRISHTI phase: spawning Spot",flush=True)
    spot=SpotFlatTerrainPolicy(prim_path="/World/Spot",position=[0,0,.8])
    command=torch.zeros(3,device="cuda")
    sim_state={"t":0.0,"initialized":False}
    def step(dt,context):
        if not sim_state["initialized"]:
            spot.initialize();sim_state["initialized"]=True
        else:spot.forward(dt,command)
        sim_state["t"]+=dt
    callback=SimulationManager.register_callback(step,IsaacEvents.POST_PHYSICS_STEP)
    advance=lambda:app.update()

    # Attach to an actual rigid body so cameras move with Spot rather than its root Xform.
    rigid=[p for p in stage.Traverse() if str(p.GetPath()).startswith("/World/Spot/") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    print("DRISHTI phase: Spot rigid bodies",[str(p.GetPath()) for p in rigid],flush=True)
    if args.body_prim:
        body=stage.GetPrimAtPath(args.body_prim)
        if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("--body-prim must name a rigid body in the Spot asset")
    else:
        preferred=[p for p in rigid if p.GetName().lower() in ("body","base","base_link","trunk")]
        if len(preferred)!=1:
            raise RuntimeError("Cannot identify camera parent safely. Supply --body-prim from: "+str([str(p.GetPath()) for p in rigid]))
        body=preferred[0]
    cameras=[]
    for name,lateral in (("left",.06),("right",-.06)):
        c=Camera(prim_path=str(body.GetPath())+f"/drishti_{name}",name=name,resolution=(640,480),frequency=20)
        # Camera accepts a ROS-frame orientation; navigation uses a separate
        # OpenCV-optical transform with the same physical mount.
        mount=camera_ros_mount(height=.15,left=lateral)
        c.set_local_pose(translation=mount[:3,3],orientation=rot_matrix_to_quat(mount[:3,:3]),camera_axes="ros")
        c.set_focal_length(2.4);c.set_horizontal_aperture(3.66)
        c.set_clipping_range(.05,60)
        cameras.append(c)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([-3,-5,3]),target=np.array([5,0,.4]))
    except ImportError:pass
    stage.GetRootLayer().Export(str(output/"scene.usda"))
    (output/"run_config.json").write_text(json.dumps({"mode":args.mode,"adapter":"isaac_6_0_1",
        "goal":[args.goal_x,args.goal_y],"body_prim":str(body.GetPath()),"sensor":"rectified stereo RGB at 20 Hz",
        "depth":"SGBM image-derived","pose":"PnP visual odometry, no loop closure",
        "launch_pad_prior":"known flat one-metre radius; nominal settled body height 0.50m",
        "locomotion":"installed NVIDIA flat-terrain policy; uses simulator proprioception internally"},indent=2))
    movie=None
    if args.cinematic:
        from cinematic import CinematicRecorder
        def presentation_position():
            # This is a rendering-only truth read, not a navigation input.
            poses=spot.robot.get_world_poses()[0]
            return poses.numpy()[0] if hasattr(poses,"numpy") else np.asarray(poses)[0]
        movie=CinematicRecorder(output,presentation_position)
    log=(output/"telemetry.jsonl").open("w")
    writer=cv2.VideoWriter(str(output/"camera_preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(1280,480))
    if not writer.isOpened():raise RuntimeError("Cannot open video encoder")
    print("DRISHTI mode:",args.mode,"Output:",output,flush=True)
    timeline=omni.timeline.get_timeline_interface()
    timeline.play();timeline.commit()
    # Let SimulationManager select its tensor backend before camera wrappers
    # create their render products. This matters on the CUDA pipeline.
    advance()
    # Installed 6.0.1 Camera examples initialize after playback starts.
    for c in cameras:c.initialize()
    if movie is not None:movie.start()
    pipeline=None;last_frame=-1.0;last_used_render=-1.0;last_valid_sim=0.;count=0
    while app.is_running() and sim_state["t"]<args.duration:
        if movie is not None and sim_state["initialized"]:
            movie.aim(sim_state["t"])
        advance();t=sim_state["t"]
        if movie is not None:
            presentation_state = ("POLICY_DEMO — NOT CAMERA NAV" if args.mode=="policy_demo"
                                  else (pipeline.navigator.state if pipeline is not None else "WARMUP"))
            movie.record(t,presentation_state)
        if t<3:continue
        frame0,frame1=[c.get_current_frame() for c in cameras]
        stamps=[f.get("rendering_time") for f in (frame0,frame1)]
        if any(s is None for s in stamps) or abs(float(stamps[0])-float(stamps[1]))>.01 or float(stamps[0])<=last_used_render:
            command[:]=0
            if count%25==0:print("HOLD: waiting for synchronized fresh camera frames",flush=True)
            count+=1;continue
        last_used_render=float(stamps[0]);last_frame=t
        images=[c.get_rgba() for c in cameras]
        if any(im is None or im.size==0 for im in images):command[:]=0;continue
        left,right=[rgb_u8(im) for im in images]
        if pipeline is None:
            K=cameras[0].get_intrinsics_matrix()
            cal=Calibration(float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]),.12,640,480)
            pipeline=CameraNavigation(cal,(args.goal_x,args.goal_y))
            (output/"calibration.json").write_text(json.dumps(cal.__dict__,indent=2))
        start=time.perf_counter()
        cmd,stats,depth=pipeline.process(left,right,last_used_render,max(0,t-last_used_render))
        stats["processing_ms"]=(time.perf_counter()-start)*1000
        if args.mode=="capture":
            cmd[:]=0;stats["state"]="CAPTURE";stats["reason"]="Stationary camera verification"
        elif args.mode=="policy_demo":
            # This deliberately bypasses the vision planner. It verifies the
            # installed policy and gives the presentation camera a moving
            # subject; telemetry labels it so it cannot be mistaken for
            # camera-only navigation.
            cmd[:]=(.30,0.,0.);stats["state"]="POLICY_DEMO";stats["reason"]="Installed policy visual verification; not camera navigation"
        command[:]=torch.as_tensor(cmd,dtype=command.dtype)
        stats["applied_command"]=cmd.tolist()
        log.write(json.dumps(stats)+"\n");log.flush()
        view=cv2.cvtColor(left,cv2.COLOR_RGB2BGR)
        d=np.nan_to_num(depth,nan=0) if depth is not None else np.zeros((480,640))
        heat=cv2.applyColorMap(np.uint8(np.clip(d/7,0,1)*255),cv2.COLORMAP_TURBO)
        heat[~np.isfinite(depth) if depth is not None else np.ones((480,640),bool)]=0
        canvas=np.hstack((view,heat))
        cv2.putText(canvas,f"DRISHTI  {stats['state']}  VO q={stats.get('quality',0):.2f}",(18,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        cv2.putText(canvas,"RGB camera                                      IMAGE-DERIVED STEREO DEPTH",(18,465),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        writer.write(canvas)
        if count%10==0:
            cv2.imwrite(str(output/"latest_camera.jpg"),canvas)
            print(json.dumps(stats),flush=True)
        if count%50==0:
            np.savez_compressed(output/f"pair_{count:05d}.npz",left=left,right=right,timestamp=last_used_render)
        count+=1
        if stats["state"]=="ARRIVED":break
    command[:]=0
    log.close();writer.release()
    if movie is not None:movie.close()
    poses=spot.robot.get_world_poses()[0]
    position=poses.numpy()[0] if hasattr(poses,"numpy") else np.asarray(poses)[0]
    (output/"run_summary.json").write_text(json.dumps({
        "mode":args.mode,"sim_seconds":sim_state["t"],"stereo_frames":count,
        "cinematic_frames":0 if movie is None else movie.frames,
        "final_spot_world_position":[float(v) for v in position[:3]],
        "truth_use":"presentation camera and post-run evaluator only; never navigation"
    },indent=2))
    print("Run ended. Review telemetry before claiming navigation success.",flush=True)
except BaseException:
    error=traceback.format_exc()
    print("DRISHTI ERROR:\n"+error,flush=True)
    (output/"failure.txt").write_text(error)
    raise
finally:
    app.close()
