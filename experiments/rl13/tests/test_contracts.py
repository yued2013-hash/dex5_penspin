import json
import numpy as np
import mujoco
import platform,pytest
from experiments.rl13.env import ROOT,ResidualPenEnv,CONFIG

def test_reference_and_model_contract():
    e=ResidualPenEnv(training=False)
    assert e.m.nu==16 and e.m.neq==4 and e.m.nmocap==0
    assert e.m.jnt_type[e.m.joint('pen_free').id]==mujoco.mjtJoint.mjJNT_FREE
    assert e.states.shape==(e.n+1,mujoco.mj_stateSize(e.m,mujoco.mjtState.mjSTATE_INTEGRATION))
    assert np.isfinite(e.states).all()

@pytest.mark.skipif(platform.system()!='Windows',reason='Golden trajectory was captured on Windows; cross-OS contact portability is not certified.')
def test_windows_zero_residual_reproduces_golden_continuations():
    e=ResidualPenEnv(training=False)
    for start in [0,500,964,1650]:
        obs,_=e.reset(options={'frame':start,'full':True})
        assert obs.shape==(101,) and e.observation_space.contains(obs)
        for _ in range(50):
            obs,_,done,_,_=e.step(np.zeros(16))
            assert not done
            # R2 golden-data contract is only certified on the recorded Windows platform.
            np.testing.assert_allclose(e.d.qpos,e.reference_q[e.k],atol=1e-6,rtol=0)

def test_same_platform_restoration_is_reproducible():
    a,b=ResidualPenEnv(training=False),ResidualPenEnv(training=False)
    for start in [0,500,964,1650]:
        a.reset(options={'frame':start,'full':True});b.reset(options={'frame':start,'full':True})
        for _ in range(50):
            left=a.step(np.zeros(16));right=b.step(np.zeros(16))
            assert left[2]==right[2]
            np.testing.assert_allclose(a.d.qpos,b.d.qpos,atol=1e-12,rtol=0)
            np.testing.assert_allclose(a.d.qvel,b.d.qvel,atol=1e-12,rtol=0)

def test_residual_authority_limits_and_no_external_forces():
    e=ResidualPenEnv(training=False);e.reset(options={'frame':1850,'full':True})
    for _ in range(10):
        e.step(np.full(16,100.))
        assert abs(e.offset).max()<=CONFIG['residual_limit_rad']
        assert np.all(e.d.ctrl<=e.m.actuator_ctrlrange[:,1]) and np.all(e.d.ctrl>=e.m.actuator_ctrlrange[:,0])
        assert not np.any(e.d.qfrc_applied) and not np.any(e.d.xfrc_applied)

def test_reference_does_not_claim_independent_generalization():
    p=json.loads((ROOT/'data/provenance.json').read_text())
    assert p['independent_source_trajectories']==1
    assert p['train_test_source_generalization_measured'] is False

def test_quality_gate_rejects_rotation_without_collision_quality():
    q=json.loads((ROOT/'results/quality_audit.json').read_text())
    assert q['baseline']['behavior_success']
    assert q['baseline']['penetration_mm']>.5
    assert q['baseline']['strict_success'] is False

def test_training_budget_matches_actual_checkpoints():
    from stable_baselines3 import PPO
    first=json.loads((ROOT/'results/training_report.json').read_text());micro=json.loads((ROOT/'results/micro_training_report.json').read_text())
    assert first['actual_training_steps']==micro['actual_training_steps']==32768
    assert PPO.load(ROOT/'checkpoints/ppo_best.zip').num_timesteps==first['selected_trained_model_steps']
    assert PPO.load(ROOT/'checkpoints/micro_best.zip').num_timesteps==micro['selected_model_steps']
