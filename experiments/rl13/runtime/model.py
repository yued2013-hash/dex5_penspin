"""Unitree Dex5-1, 16 actuators and four coupled distal joints.

URDF geometry/inertia from Unitree; PD gains/friction are simulation assumptions.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import sys,json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
ACTIVE=['Yaw_11R','Roll_12R','Pitch_13R','Pitch_14R']+ [f'{kind}_{i}{j}R' for i in range(2,6) for kind,j in [('Roll',1),('Pitch',2),('Pitch',3)]]
ALL=['Yaw_11R','Roll_12R','Pitch_13R','Pitch_14R']+[f'{kind}_{i}{j}R' for i in range(2,6) for kind,j in [('Roll',1),('Pitch',2),('Pitch',3),('Pitch',4)]]
RATIO=1.396263378/1.684242699

def hand_spec(coupled=True):
    folder=ROOT/'assets/hand'
    tree=ET.parse(folder/'Dex5-URDF-R.urdf').getroot()
    compiler=tree.find('mujoco/compiler');compiler.set('meshdir','');compiler.set('fusestatic','false')
    for e in tree.findall('.//mesh'):e.set('filename',Path(e.get('filename')).name)
    assets={p.name:p.read_bytes() for p in (folder/'meshes').glob('*.STL')}
    spec=mujoco.MjSpec.from_string(ET.tostring(tree,encoding='unicode'),assets=assets)
    for i,g in enumerate(spec.geoms):
        g.name='geom_'+str(i)
        if g.contype or g.conaffinity:
            g.group=3;g.condim=4;g.priority=1;g.friction=[.7,.005,.0001]
            g.solref=[.015,1.2];g.solimp=[.9,.95,.001,.5,2]
        else:g.group=1;g.rgba[3]=1
    for j in spec.joints:
        j.armature=.00012;j.damping[0]=.003
    for name in ACTIVE:
        joint=spec.joint(name)
        torque=.93 if '11R' in name or '12R' in name or '13R' in name or '14R' in name else (.22 if 'Roll' in name else .5)
        a=spec.add_actuator(name=name+'_act',target=name,trntype=mujoco.mjtTrn.mjTRN_JOINT,gaintype=mujoco.mjtGain.mjGAIN_FIXED,biastype=mujoco.mjtBias.mjBIAS_AFFINE)
        a.gainprm[0]=3.;a.biasprm[1]=-3.;a.biasprm[2]=-.08
        a.forcelimited=1;a.forcerange=[-torque,torque]
        a.ctrllimited=1;a.ctrlrange=joint.range
    if coupled:
        for i in range(2,6):
            spec.add_equality(name=f'distal_coupling_{i}',type=mujoco.mjtEq.mjEQ_JOINT,objtype=mujoco.mjtObj.mjOBJ_JOINT,name1=f'Pitch_{i}4R',name2=f'Pitch_{i}3R',data=[0,RATIO,0,0,0,0,0,0,0,0,0,0],solref=[.005,1],solimp=[.99,.999,.001,.5,2])
    # Tip/knuckle sites support geometric retargeting and diagnostics.
    for i in range(1,6):
        tip=[.052,.004,0] if i==1 else [.025,-.006,0]
        spec.body(f'Link_{i}4R').add_site(name=f'tip{i}',pos=tip,size=[.002,0,0],group=4)
        spec.body(f'Link_{i}3R').add_site(name=f'pip{i}',pos=[0,0,0],size=[.002,0,0],group=4)
        for point in ['tip','pip']:
            spec.add_sensor(name=f'{point}_position_{i}',type=mujoco.mjtSensor.mjSENS_FRAMEPOS,objtype=mujoco.mjtObj.mjOBJ_SITE,objname=f'{point}{i}')
    # Base and first-link assembly overlaps are intentional, not finger contacts.
    for i in range(1,6):
        spec.add_exclude(bodyname1='base_link00',bodyname2=f'Link_{i}1R')
        spec.add_exclude(bodyname1='base_link00',bodyname2=f'Link_{i}2R')
    return spec

def build(aligned=True,pen=True):
    spec=hand_spec()
    spec.option.timestep=.001;spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations=50;spec.option.impratio=10
    spec.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.solver=mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.tolerance=1e-10
    if aligned:
        align=json.loads((ROOT/'assets/alignment.json').read_text())
        spec.body('base_link00').pos=align['position'];spec.body('base_link00').quat=align['quaternion']
    spec.visual.global_.offwidth=1280;spec.visual.global_.offheight=720
    spec.worldbody.add_light(pos=[0,-1,2],dir=[0,0,-1],diffuse=[.9,.9,.9])
    spec.worldbody.add_light(pos=[1,1,1.5],dir=[-1,-1,-1],diffuse=[.6,.65,.8])
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2,2,.01],rgba=[.16,.2,.25,1],group=0)
    if pen:
        from .generate_pen import generate
        dest=generate()
        assets=spec.assets.copy()
        for name in ['shaft','endcap']:
            spec.add_mesh(name='pen_'+name,file=name+'.obj');assets[name+'.obj']=(dest/(name+'.obj')).read_bytes()
        spec.assets=assets
        b=spec.worldbody.add_body(name='pen',pos=[0,0,.6]);b.add_freejoint(name='pen_free')
        for name,pos,rgba in [('shaft',[0,0,0],[.95,.7,.12,1]),('endcap',[-.107,0,0],[.08,.45,.95,1]),('endcap',[.107,0,0],[.95,.18,.12,1])]:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname='pen_'+name,pos=pos,rgba=rgba,contype=0,conaffinity=0,group=1,density=0)
        b.add_geom(name='shaft_collision',type=mujoco.mjtGeom.mjGEOM_CYLINDER,fromto=[-.089,0,0,.089,0,0],size=[.00745,0,0],density=469.05012453696503,group=3,contype=1,conaffinity=2)
        for i,x in enumerate([-.107,.107]):b.add_geom(name=f'cap{i}_collision',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,0,0],size=[.018,.009,.009],density=469.05012453696503,group=3,contype=1,conaffinity=2)
        # Instrumentation only: contact centroid and world force on the pen.
        # XBODY selects a subtree (middle + distal phalanges), BODY a single body.
        for i in range(1,6):
            spec.add_sensor(name=f'finger{i}_pen_contact',type=mujoco.mjtSensor.mjSENS_CONTACT,objtype=mujoco.mjtObj.mjOBJ_XBODY,objname=f'Link_{i}3R',reftype=mujoco.mjtObj.mjOBJ_BODY,refname='pen',intprm=[19,3,1])
        spec.add_sensor(name='palm_pen_contact',type=mujoco.mjtSensor.mjSENS_CONTACT,objtype=mujoco.mjtObj.mjOBJ_BODY,objname='base_link00',reftype=mujoco.mjtObj.mjOBJ_BODY,refname='pen',intprm=[19,3,1])
    model=spec.compile()
    if pen:
        bid=model.body('pen').id;model.body_inertia[bid]*=.03/model.body_mass[bid];model.body_mass[bid]=.03
    return model

def set_angles(model,data,active):
    for i,n in enumerate(ACTIVE):data.qpos[model.jnt_qposadr[model.joint(n).id]]=active[i]
    for i in range(2,6):data.qpos[model.jnt_qposadr[model.joint(f'Pitch_{i}4R').id]]=RATIO*active[4+(i-2)*3+2]

if __name__=='__main__':
    m=build(aligned=False);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    print('nq',m.nq,'nu',m.nu,'neq',m.neq)
