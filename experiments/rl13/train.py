"""Bounded one-seed PPO smoke. A trained candidate is never silently promoted."""
import argparse,json,time,hashlib
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .env import ROOT,CONFIG,ResidualPenEnv
from .evaluate import evaluate

class AuditCallback(BaseCallback):
    def __init__(self):
        super().__init__();self.next=8192;self.best=-float('inf');self.started=time.time()
    def _on_step(self):
        if self.num_timesteps>=self.next:
            self.next+=8192;tag=f'ppo_{self.num_timesteps}';r=evaluate(self.model,tag,save=True);r.update(training_steps=int(self.num_timesteps),wall_seconds=time.time()-self.started)
            with (ROOT/'results/training.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            self.model.save(ROOT/'checkpoints'/tag)
            if r['selection_score']>self.best:
                self.best=r['selection_score'];self.model.save(ROOT/'checkpoints/ppo_best')
            print(json.dumps(r),flush=True)
        return True

def make_env(rank):return lambda:ResidualPenEnv(training=True,rank=rank)

def main(steps=None):
    torch.set_num_threads(1);gate=json.loads((ROOT/'results/reset_validation.json').read_text());assert gate['ready_for_bounded_training'];assert not (ROOT/'results/training.jsonl').exists(),'Use a new versioned experiment for another run.'
    baseline=evaluate(tag='baseline');print('BASELINE',json.dumps(baseline),flush=True)
    env=SubprocVecEnv([make_env(i) for i in range(CONFIG['num_envs'])],start_method='spawn')
    model=PPO('MlpPolicy',env,learning_rate=CONFIG['learning_rate'],n_steps=CONFIG['n_steps'],batch_size=CONFIG['batch_size'],n_epochs=CONFIG['n_epochs'],gamma=CONFIG['gamma'],gae_lambda=CONFIG['gae_lambda'],clip_range=CONFIG['clip_range'],ent_coef=0.,policy_kwargs=dict(net_arch=dict(pi=CONFIG['net_arch'],vf=CONFIG['net_arch']),log_std_init=CONFIG['log_std_init']),seed=CONFIG['seed'],device='cpu',verbose=0)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    started=time.time();callback=AuditCallback();model.learn(total_timesteps=steps or CONFIG['total_timesteps'],callback=callback);model.save(ROOT/'checkpoints/ppo_last');env.close()
    best=PPO.load(ROOT/'checkpoints/ppo_best');trained=evaluate(best,'trained_best');promoted=bool(trained['success'] and trained['selection_score']>baseline['selection_score'])
    report=dict(phase=CONFIG['phase'],seed=CONFIG['seed'],actual_training_steps=int(model.num_timesteps),selected_trained_model_steps=int(best.num_timesteps),wall_seconds=time.time()-started,baseline=baseline,trained_candidate=trained,trained_candidate_promoted=promoted,recommended='trained_best' if promoted else 'baseline',observation_dim=101,action_dim=16,reference_sha256=hashlib.sha256((ROOT/'data/reference.npz').read_bytes()).hexdigest(),end_to_end_from_scratch=False,hardware_commands_issued=False,independent_test_sources=0)
    (ROOT/'results/training_report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int);a=p.parse_args();main(a.steps)
