"""Continuous physical evaluation; pose-derived rotation, contact and capture gates."""
import argparse,json
import mujoco,numpy as np
from .env import ROOT,CONFIG,ResidualPenEnv
from .runtime.contact_metrics import axis,contacts

def evaluate(policy=None, tag='baseline', strength=1., save=True, residual_limit=None):
    e=ResidualPenEnv(training=False,residual_limit=residual_limit);e.strength=strength;obs,_=e.reset(options={'full':True})
    m,d=e.m,e.d;u=axis(d.qpos[e.pq+3:e.pq+7]);u-=e.normal*(u@e.normal);u/=np.linalg.norm(u);v=np.cross(e.normal,u)
    target=e.reference_q[-1];rows=[];states=[];controls=[];actions=[];observations=[];forces=[];external=0.;gap=longest=0;total_reward=0.
    previous=np.arctan2(axis(d.qpos[e.pq+3:e.pq+7])@v,axis(d.qpos[e.pq+3:e.pq+7])@u);turn=0.;oldquat=d.qpos[e.pq+3:e.pq+7].copy()
    initial=e.states[0].copy();states.append(initial)
    for k in range(e.n):
        observations.append(obs.copy());action=policy.predict(obs,deterministic=True)[0] if policy is not None else np.zeros(16)
        obs,reward,terminated,truncated,info=e.step(action);total_reward+=reward
        pose=d.qpos[e.pq:e.pq+7];direction=axis(pose[3:]);angle=np.arctan2(direction@v,direction@u);turn+=(angle-previous+np.pi)%(2*np.pi)-np.pi;previous=angle
        qerr=d.qpos[e.qids]-target[e.qids];phi=np.zeros(3);mujoco.mju_subQuat(phi,pose[3:],oldquat);oldquat=pose[3:].copy();poseomega=float(np.linalg.norm(phi)/.02)
        c=contacts(m,d);force=e.forces();forces.append(force);gap=gap+1 if not c['loaded_fingers'] else 0;longest=max(gap,longest)
        penetration=max([max(0,-x['distance_m'])*1000 for x in c['contacts']]+[0.])
        rows.append(dict(time=(k+1)*.02,turn_deg=float(np.degrees(turn)),position_mm=float(np.linalg.norm(pose[:3]-target[e.pq:e.pq+3])*1000),full_angle_deg=float(np.degrees(2*np.arccos(np.clip(abs(pose[3:]@target[e.pq+3:e.pq+7]),0,1)))) ,joint_rms=float(np.sqrt(np.mean(qerr*qerr))),joint_max=float(abs(qerr).max()),pinch=c['thumb_index_or_middle_pinch'],loaded=len(c['loaded_fingers']),palm=c['palm_contact'],penetration_mm=penetration,projection=float(np.hypot(direction@u,direction@v)),linear_speed=float(np.linalg.norm(d.qvel[e.pv:e.pv+3])),angular_speed_pose=poseomega,angular_speed_instantaneous=float(np.linalg.norm(d.qvel[e.pv+3:e.pv+6])),reference_position_error_mm=info['reference_position_error_mm'],failure=info['failure']))
        state=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);states.append(state);controls.append(d.ctrl.copy());actions.append(action)
        external=max(external,float(abs(d.qfrc_applied).max()),float(abs(d.xfrc_applied).max()))
        if terminated or truncated:break
    late=rows[-250:];g=CONFIG['terminal_grasp'];qvel=np.asarray(states)[:,1+m.nq:1+m.nq+m.nv][:,e.vids];acc=np.linalg.norm(np.diff(qvel,axis=0)/.02,axis=1);delta=np.linalg.norm(np.diff(controls,axis=0),axis=1)
    metrics=dict(position_mm=max(r['position_mm'] for r in late),angle_deg=max(r['full_angle_deg'] for r in late),joint_rms_rad=max(r['joint_rms'] for r in late),joint_max_rad=max(r['joint_max'] for r in late),pinch_fraction=float(np.mean([r['pinch'] for r in late])),linear_speed_p95=float(np.percentile([r['linear_speed'] for r in late],95)),angular_speed_pose_p95=float(np.percentile([r['angular_speed_pose'] for r in late],95)))
    complete=len(rows)==e.n;capture=bool(complete and metrics['position_mm']<g['position_mm'] and metrics['angle_deg']<g['angle_deg'] and metrics['joint_rms_rad']<g['joint_rms_rad'] and metrics['joint_max_rad']<g['joint_max_rad'] and metrics['pinch_fraction']>=g['pinch_fraction'] and metrics['linear_speed_p95']<g['linear_speed_p95'] and metrics['angular_speed_pose_p95']<g['angular_speed_p95'])
    # A release/recontact event must last >=3 consecutive 20ms control steps on both sides.
    events=[];f=np.asarray(forces)>.03
    for finger in range(5):
        stable=bool(f[0,finger]);pending=0;released=False
        for k in range(1,len(f)):
            if bool(f[k,finger])!=stable:
                pending+=1
                if pending==3:
                    stable=not stable;pending=0
                    if not stable:released=True
                    elif released:
                        events.append(dict(finger=finger+1,time=rows[k]['time'],support_at_recontact=rows[k]['loaded']>=2));released=False
            else:pending=0
    success=bool(capture and max(r['turn_deg'] for r in rows)>=355 and min(r['projection'] for r in rows)>.4 and not any(r['palm'] or r['failure'] for r in rows) and not sum(w.number for w in d.warning) and longest*.02<=.1 and external==0)
    summary=dict(tag=tag,source_id=CONFIG['reference_source_id'],episodes=1,initial_state='fixed_true_initial_state',independent_test_sources=0,complete=complete,success=success,capture_passed=capture,duration_s=len(rows)*.02,max_turn_deg=max(r['turn_deg'] for r in rows),terminal_last5=metrics,whole_episode_joint_acceleration_mean=float(acc.mean()),whole_episode_command_delta_mean=float(delta.mean()),maximum_penetration_mm=max(r['penetration_mm'] for r in rows),palm=any(r['palm'] for r in rows),warnings=int(sum(w.number for w in d.warning)),longest_unloaded_s=longest*.02,external_force_max=external,debounced_release_recontact_events=len(events),supported_recontact_events=sum(x['support_at_recontact'] for x in events),debounce_steps=3,total_reward=total_reward,strength=strength,scope='single-source deterministic simulation; not a generalization success rate')
    summary['selection_score']=10000*success+100*complete-summary['terminal_last5']['position_mm']-summary['terminal_last5']['angle_deg']-100*(1-summary['terminal_last5']['pinch_fraction'])-summary['whole_episode_joint_acceleration_mean']
    summary['residual_limit_rad']=e.residual_limit
    summary['physical_quality_passed']=bool(summary['maximum_penetration_mm']<=.5)
    summary['strict_success']=bool(success and summary['physical_quality_passed'])
    summary['behavior_success']=summary['success']
    summary['success_definition']='rotation and terminal capture; strict_success additionally requires <=0.5mm mesh penetration'
    if save:
        (ROOT/'results'/f'{tag}.json').write_text(json.dumps(dict(summary=summary,trace=rows,gait_events=events),indent=2))
        np.savez_compressed(ROOT/'data'/f'{tag}_rollout.npz',states=states,ctrl=controls,actions=actions,observations=observations,normal=e.normal,dt=.02)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--tag',default='baseline');p.add_argument('--strength',type=float,default=1);a=p.parse_args();policy=None
    if a.checkpoint:
        from stable_baselines3 import PPO
        policy=PPO.load(a.checkpoint)
    print(json.dumps(evaluate(policy,a.tag,a.strength),indent=2))
