"""Matched-budget residual-authority ablation; preserves the initial smoke bytes."""
import json,time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from .env import ROOT,CONFIG,ResidualPenEnv
from .evaluate import evaluate
LIMIT=.00006

def factory(rank):return lambda:ResidualPenEnv(rank=rank,residual_limit=LIMIT)

class Audit(BaseCallback):
    def __init__(self):super().__init__();self.next=8192;self.best=-float('inf');self.started=time.time()
    def _on_step(self):
        if self.num_timesteps>=self.next:
            self.next+=8192;r=evaluate(self.model,f'micro_{self.num_timesteps}',residual_limit=LIMIT);r.update(training_steps=int(self.num_timesteps),wall_seconds=time.time()-self.started)
            with (ROOT/'results/micro_training.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            self.model.save(ROOT/f'checkpoints/micro_{self.num_timesteps}')
            if r['selection_score']>self.best:self.best=r['selection_score'];self.model.save(ROOT/'checkpoints/micro_best')
            print(json.dumps(r),flush=True)
        return True

def main():
    assert not (ROOT/'results/micro_training.jsonl').exists()
    torch.set_num_threads(1);env=SubprocVecEnv([factory(i) for i in range(4)],start_method='spawn')
    model=PPO('MlpPolicy',env,learning_rate=CONFIG['learning_rate'],n_steps=256,batch_size=256,n_epochs=5,gamma=.995,gae_lambda=.95,clip_range=.15,ent_coef=0,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),log_std_init=-2.5),seed=CONFIG['seed'],device='cpu',verbose=0)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    started=time.time();model.learn(total_timesteps=32768,callback=Audit());model.save(ROOT/'checkpoints/micro_last');env.close();best=PPO.load(ROOT/'checkpoints/micro_best');result=evaluate(best,'micro_best',residual_limit=LIMIT)
    report=dict(phase='RL13_micro_residual_authority_ablation',seed=CONFIG['seed'],actual_training_steps=int(model.num_timesteps),selected_model_steps=int(best.num_timesteps),residual_limit_rad=LIMIT,wall_seconds=time.time()-started,matched_to='RL13 initial smoke: all hyperparameters, starts, reward, reference identical except residual authority',result=result,hardware_commands_issued=False)
    (ROOT/'results/micro_training_report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
