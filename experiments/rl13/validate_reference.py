"""MuJoCo-specific R0–R3 checks. These do not certify PhysX or hardware resets."""
import hashlib,json
import numpy as np,mujoco
from .env import ROOT,CONFIG,ResidualPenEnv

def main():
    e = ResidualPenEnv(training=False)
    finite = bool(np.isfinite(e.states).all() and np.isfinite(e.commands).all())
    assert e.states.shape == (e.n + 1, mujoco.mj_stateSize(e.m, mujoco.mjtState.mjSTATE_INTEGRATION))
    assert e.commands.shape == (e.n,16)
    rows = []
    for frame in CONFIG['training_start_frames']:
        e.reset(options={'frame':frame, 'full':True})
        pen_body=e.m.body('pen').id
        depth=max([max(0.,-c.dist)*1000 for c in e.d.contact if pen_body in (e.m.geom_bodyid[c.geom1],e.m.geom_bodyid[c.geom2])]+[0.])
        joints=e.m.actuator_trnid[:,0];q=e.d.qpos[e.qids]
        range_error=max(0.,float(np.max(e.m.jnt_range[joints,0]-q)),float(np.max(q-e.m.jnt_range[joints,1])))
        # Explicit simulation-only reset tolerances; stricter trajectory quality remains separate.
        r0=finite and depth<=1.0 and range_error<=.01
        errors=[]
        for j in range(min(50,e.n-frame)):
            _,_,done,_,_=e.step(np.zeros(16));errors.append(float(abs(e.d.qpos-e.reference_q[e.k]).max()))
            if done:break
        continuation = max(errors) < 1e-6 and len(errors) == min(50,e.n-frame)
        e.reset(options={'frame':frame, 'full':True})
        # Bounded action-response readiness, not an arbitrary new reset distribution.
        safe=True
        for j in range(min(10,e.n-frame)):
            _,_,done,_,_=e.step(np.full(16,.02*(-1 if j%2 else 1)))
            safe &= not done
        rows.append(dict(frame=frame,R0_finite=finite,R0_bounded_geometry=r0,pen_penetration_mm=depth,soft_joint_limit_excess_rad=range_error,R1_stable=continuation,R2_continuation=continuation,
                         R3_bounded_action_ready=bool(safe),maximum_qpos_continuation_error=max(errors)))
    approved=[x['frame'] for x in rows if x['R0_bounded_geometry'] and x['R1_stable'] and x['R2_continuation'] and x['R3_bounded_action_ready']]
    report=dict(schema='rl13-mujoco-reset-validation-v2',ready_for_bounded_training=bool(approved),approved_frames=approved,
                reset_penetration_limit_mm=1.,reset_soft_joint_limit_tolerance_rad=.01,post_training_geometry_supplement=True,
                source_sha256=hashlib.sha256((ROOT/'data/reference.npz').read_bytes()).hexdigest(),
                simulator='MuJoCo',perturbed_reset_states_authorized=False,hardware_ready=False,rows=rows)
    (ROOT/'results/reset_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert 0 in approved,'True initial state must pass.'
if __name__=='__main__':main()
