"""Re-run zero residual and real selected networks, compare with archived evidence."""
import json
import numpy as np,torch
from stable_baselines3 import PPO
from .env import ROOT
from .evaluate import evaluate

def main():
    torch.set_num_threads(1);checks=[]
    for tag,model,limit in [('baseline',None,None),('trained_best','ppo_best.zip',None),('micro_best','micro_best.zip',.00006)]:
        old=np.load(ROOT/f'data/{tag}_rollout.npz');expected=old['ctrl'].copy();q=old['states'][:,1:54].copy();old.close()
        policy=PPO.load(ROOT/'checkpoints'/model) if model else None
        summary=evaluate(policy,'verification_'+tag,residual_limit=limit)
        current=np.load(ROOT/f'data/verification_{tag}_rollout.npz');assert np.array_equal(expected,current['ctrl']);assert np.array_equal(q,current['states'][:,1:54])
        checks.append(dict(tag=tag,exact_controls=True,exact_qpos_qvel=True,complete=summary['complete'],behavior_success=summary['behavior_success'],strict_success=summary['strict_success']))
    report=dict(simulator='MuJoCo',independent_real_network_replay=True,checks=checks)
    (ROOT/'results/replay_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
