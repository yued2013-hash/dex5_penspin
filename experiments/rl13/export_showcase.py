"""Generate static viewer data directly from measured artifacts."""
import json,hashlib
from pathlib import Path
from .env import ROOT

def main():
    repo=ROOT.parents[1];show=repo/'docs/showcase';read=lambda p:json.loads(p.read_text())
    baseline=read(ROOT/'results/baseline.json');first=read(ROOT/'results/training_report.json');micro=read(ROOT/'results/micro_training_report.json');student=read(repo/'Dex5_Sim2Real_S0/student_s0_11/student_closed_loop_eval.json')
    b=baseline['summary'];b.update(physical_quality_passed=b['maximum_penetration_mm']<=.5,strict_success=b['success'] and b['maximum_penetration_mm']<=.5)
    training=[]
    for key,file in [('large','training.jsonl'),('micro','micro_training.jsonl')]:
        for line in (ROOT/'results'/file).read_text().splitlines():
            row=json.loads(line);training.append(dict(x=row['training_steps'],**{key:row['total_reward']}))
    quality={}
    for name in ['baseline','trained_best','micro_best']:
        value=read(ROOT/f'results/{name}.json')['summary'];quality[name]=dict(behavior_success=value['success'],penetration_mm=value['maximum_penetration_mm'],maximum_allowed_penetration_mm=.5,strict_success=value['success'] and value['maximum_penetration_mm']<=.5)
    (ROOT/'results/quality_audit.json').write_text(json.dumps(quality,indent=2))
    data=dict(baseline=b,latest=micro['result'],total_training_steps=first['actual_training_steps']+micro['actual_training_steps'],promoted=False,
              student=dict(successes=student['successes'],episodes=student['episodes'],unique_sources=student['unique_source_trajectories']),
              trace=[dict(x=r['time'],turn=r['turn_deg'],pinch=int(r['pinch'])) for r in baseline['trace'][::5]],training_series=training,
              reference_sha256=hashlib.sha256((ROOT/'data/reference.npz').read_bytes()).hexdigest(),quality_audit=quality)
    text=json.dumps(data,ensure_ascii=False,indent=2);(show/'data/overview.json').write_text(text,encoding='utf-8');(show/'data/overview.js').write_text('window.DEX5_DATA = '+text+';\n',encoding='utf-8')
    print('Exported',len(data['trace']),'actual trajectory points and',len(training),'training checkpoints.')
if __name__=='__main__':main()
