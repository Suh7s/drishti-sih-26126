"""Train on procedurally rendered RGB/depth patches, split by scene seed.

Labels are material/geometry identities in the renderer, never inferred from the
feature values. Held-out results only measure this synthetic domain.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .perception import PerceptionModel,extract_patch_features,DEFAULT_WEIGHTS_PATH


def render_patch(label,rng,size=24):
    yy,xx=np.indices((size,size),dtype=float)
    z0=rng.uniform(.8,5)
    # Rectified pinhole rays intersect a plane with a variable normal.
    normal=np.array([rng.uniform(-.15,.15),rng.uniform(.05,.5),1.0])
    if label==1:normal[:2]=rng.uniform(-.8,.8,2)
    depth=(z0/(1+normal[0]*(xx-size/2)/100+normal[1]*(yy-size/2)/100)).astype(np.float32)
    palettes=[(110,89,65),(115,115,110),(45,55,60),(50,140,45)]
    color=np.array(palettes[label],float)*rng.uniform(.55,1.45)
    color+=rng.normal(0,9,3)
    energy=[rng.uniform(6,18),rng.uniform(12,32),rng.uniform(.3,2.5),rng.uniform(8,27)][label]
    texture=rng.normal(0,energy,(size,size))
    texture+=cv2.resize(rng.normal(0,energy,(4,4)).astype(np.float32),(size,size))
    rgb=np.clip(color[None,None,:]+texture[:,:,None],0,255).astype(np.uint8)
    if label==1 and rng.random()<.7:
        rgb[:,size//2:]=np.clip(rgb[:,size//2:].astype(float)*.55,0,255).astype(np.uint8)
        depth[:,size//2:]+=.4
    if label==2:
        depth[rng.random(depth.shape)<rng.uniform(.35,.9)]=np.nan
    else:
        depth+=rng.normal(0,.003,depth.shape).astype(np.float32)
        depth[rng.random(depth.shape)<.02]=np.nan
    return rgb,depth


def generate_synthetic_dataset(num_samples=6000,seed=26126):
    rng=np.random.default_rng(seed);features=[];labels=[]
    for i in range(num_samples):
        label=i%4;rgb,depth=render_patch(label,rng)
        features.append(extract_patch_features(rgb,depth));labels.append(label)
    return np.asarray(features,np.float32),np.asarray(labels,np.int64)


def train(weights_out=DEFAULT_WEIGHTS_PATH,epochs=40,lr=.003,batch_size=128):
    X,y=generate_synthetic_dataset(8000,26126)
    V,vy=generate_synthetic_dataset(2000,926126)
    mean=X.mean(0);scale=np.maximum(X.std(0),.05);X=(X-mean)/scale;V=(V-mean)/scale
    model=PerceptionModel(weights_path=None);model.mean=mean;model.scale=scale
    rng=np.random.default_rng(42)
    names=['W1','b1','W2','b2','W3','b3'];velocity={k:np.zeros_like(getattr(model,k)) for k in names}
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(X)),int(np.ceil(len(X)/batch_size))):
            x=X[idx];target=y[idx]
            a1=x@model.W1+model.b1;h1=np.maximum(a1,0);a2=h1@model.W2+model.b2;h2=np.maximum(a2,0)
            logits=h2@model.W3+model.b3;p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            p[np.arange(len(idx)),target]-=1;p/=len(idx)
            g3=h2.T@p;gb3=p.sum(0);d2=(p@model.W3.T)*(a2>0)
            g2=h1.T@d2;gb2=d2.sum(0);d1=(d2@model.W2.T)*(a1>0)
            gradients=[x.T@d1,d1.sum(0),g2,gb2,g3,gb3]
            for name,grad in zip(names,gradients):
                velocity[name]=.9*velocity[name]-lr*np.clip(grad,-5,5)
                setattr(model,name,getattr(model,name)+velocity[name])
    model.save_weights(weights_out)
    # Inference input is unnormalised, as in the online extractor.
    Vraw=V*scale+mean;start=time.perf_counter();prob=model.forward(Vraw);latency=(time.perf_counter()-start)*1000
    predictions=prob.argmax(1);cm=np.zeros((4,4),int);np.add.at(cm,(vy,predictions),1)
    report={'training_samples':len(X),'held_out_samples':len(V),'train_scene_seed':26126,'held_out_scene_seed':926126,
            'feature_count':40,'architecture':[40,64,32,4],'confusion_matrix_rows_true':cm.tolist(),
            'held_out_accuracy':float((predictions==vy).mean()),'batch_inference_2000_ms':latency,
            'scope':'Procedurally rendered RGB/depth material patches. Not real flood-water detection accuracy.',
            'feature_extraction_included_in_latency':False}
    Path(weights_out).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return model

if __name__=='__main__':train()
