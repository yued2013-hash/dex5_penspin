"""True pen-to-finger contacts, forces, and opposing radial grasp geometry."""
import re
import numpy as np
import mujoco

def axis(q):
    mat=np.zeros(9);mujoco.mju_quat2Mat(mat,np.asarray(q,dtype=float));return mat.reshape(3,3)[:,0]

def contacts(model,data):
    penid=model.body('pen').id;pq=int(model.jnt_qposadr[model.joint('pen_free').id]);pose=data.qpos[pq:pq+7];direction=axis(pose[3:]);rows=[]
    for k,c in enumerate(data.contact):
        if model.geom_bodyid[c.geom1]==penid:g=int(c.geom2)
        elif model.geom_bodyid[c.geom2]==penid:g=int(c.geom1)
        else:continue
        body=model.body(model.geom_bodyid[g]).name;match=re.fullmatch(r'Link_([1-5])([1-4])R',body)
        force=np.zeros(6);mujoco.mj_contactForce(model,data,k,force)
        offset=c.pos-pose[:3];radial=offset-direction*(offset@direction);radial/=max(1e-10,np.linalg.norm(radial))
        rows.append({'body':body,'finger':int(match.group(1)) if match else 0,'segment':int(match.group(2)) if match else 0,'normal_force_N':float(max(0,force[0])),'position':c.pos.copy().tolist(),'radial':radial.tolist(),'distance_m':float(c.dist)})
    fingers={i:sum(r['normal_force_N'] for r in rows if r['finger']==i) for i in range(1,6)}
    tips={i:sum(r['normal_force_N'] for r in rows if r['finger']==i and r['segment']>=3) for i in range(1,6)}
    opposed=[]
    for a in rows:
        if a['normal_force_N']<.01:continue
        for b in rows:
            if b['finger']<=a['finger'] or b['normal_force_N']<.01:continue
            dot=float(np.asarray(a['radial'])@np.asarray(b['radial']))
            if dot<-.5:opposed.append((a['finger'],b['finger'],dot))
    pinch=bool(tips[1]>.03 and (tips[2]>.03 or tips[3]>.03) and any(a==1 and b in [2,3] for a,b,_ in opposed))
    return {'contacts':rows,'fingers_force_N':fingers,'distal_force_N':tips,'opposed_pairs':opposed,'thumb_index_or_middle_pinch':pinch,'palm_contact':any(r['finger']==0 for r in rows),'loaded_fingers':[i for i in fingers if fingers[i]>.02]}
