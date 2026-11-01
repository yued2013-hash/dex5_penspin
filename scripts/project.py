"""Single presentation/evidence entry point; preserves historical training interfaces."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['showcase','evidence','audit']);a,extra=p.parse_known_args()
 if a.command=='evidence':
  project=json.loads((ROOT/'project.json').read_text(encoding='utf-8'));report=json.loads((ROOT/project['student_report']).read_text());print(json.dumps(dict(project=project,student=dict(successes=report['successes'],episodes=report['episodes'],unique_sources=report['unique_source_trajectories']),scope='Student短角度仿真；视频为独立标注的项目诊断素材'),ensure_ascii=False,indent=2));return
 script='serve_showcase.py' if a.command=='showcase' else 'audit_showcase.py'
 raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts'/script),*extra],cwd=ROOT))
if __name__=='__main__':main()
