/* Static, offline-capable viewer. All values are exported from actual reports. */
const data = window.DEX5_DATA;
const $ = (s) => document.querySelector(s);
const video = $('#demo-video');
$('#play-hero').addEventListener('click', () => { video.play().catch(() => {}); video.scrollIntoView({behavior:'smooth',block:'center'}); });
document.querySelectorAll('[data-time]').forEach(button => button.addEventListener('click', () => {video.currentTime=Number(button.dataset.time);video.play().catch(()=>{});}));
$('#metric-turn').innerHTML = `${data.baseline.max_turn_deg.toFixed(1)}<span>°</span>`;
$('#metric-depth').innerHTML = `${data.baseline.maximum_penetration_mm.toFixed(2)}<span>mm</span>`;
$('#metric-steps').innerHTML = `${data.total_training_steps.toLocaleString()}<span>步</span>`;
const fmt = (x, digits=1) => Number(x).toFixed(digits);
function resultTab(name) {
 const b=data.baseline, r=data.latest, s=data.student;
 const blocks={
 mujoco:{title:'整圈转动 + 稳定夹持，几何质量仍需优化',badge:'参考辅助 · 单轨迹',description:'物理规划前缀与既有PPO收尾组成基准。整圈转动和末段夹持已观察到；全程最大穿透超过0.5mm严格门槛，因此不标记为严格任务成功。',values:[['最大实际转角',fmt(b.max_turn_deg)+'°'],['末5秒夹持',fmt(b.terminal_last5.pinch_fraction*100,0)+'%'],['全程穿透峰值',fmt(b.maximum_penetration_mm,2)+' mm'],['独立参考来源','1 条']],sources:[['完整评估','../../experiments/rl13/results/baseline.json'],['基准视频','media/baseline-v12.mp4']]},
 rl:{title:'全程残差PPO：真实训练，按评估决定是否晋级',badge:data.promoted?'候选通过行为对比':'候选未晋级',description:`两次独立训练各32,768步，相同起点、奖励与网络，比较0.006和0.00006 rad残差上限。下方为最新小幅度消融的所选检查点；它${r.complete?'完成了全程':'未完成全程'}，${r.strict_success?'通过':'未通过'}严格物理质量门槛。`,values:[['本轮实际训练',data.total_training_steps.toLocaleString()+' 步'],['最新候选转角',fmt(r.max_turn_deg)+'°'],['实际执行时长',fmt(r.duration_s,2)+' s'],['严格任务成功',r.strict_success?'是':'否']],sources:[['初始训练日志','../../experiments/rl13/results/training.jsonl'],['幅度消融报告','../../experiments/rl13/results/micro_training_report.json'],['策略代码','../../experiments/rl13/env.py']]},
 student:{title:'Oracle → Student：历史短角度迁移基线',badge:'历史记录 · Isaac Lab',description:'原项目提供的S0.11非特权学生策略报告。任务是短角度转动与150ms保持，不能与MuJoCo整圈实验横向合并；本轮未在本机重新运行Isaac Lab。',values:[['严格短角度成功',`${s.successes} / ${s.episodes}`],['达到30°轨迹',`${s.rotation30} / ${s.episodes}`],['源隔离测试轨迹来源',String(s.unique_sources)],['硬件闭环部署','未验证']],sources:[['原始评估','../../Dex5_Sim2Real_S0/student_s0_11/student_closed_loop_eval.json'],['历史训练报告','../../Dex5_Sim2Real_S0/student_s0_11/student_training_report.json']]}
 };
 const x=blocks[name];$('#result-content').innerHTML=`<div class="result-header"><h3>${x.title}</h3><span class="badge ${name==='mujoco'||!data.promoted?'caution':''}">${x.badge}</span></div><p class="result-description">${x.description}</p><div class="result-grid">${x.values.map(([label,value])=>`<div><small>${label}</small><b>${value}</b></div>`).join('')}</div><div class="result-source">${x.sources.map(([label,url])=>`<a href="${url}">${label} ↗</a>`).join('')}</div>`;
 document.querySelectorAll('[data-tab]').forEach(button=>{const active=button.dataset.tab===name;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));});
}
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>resultTab(button.dataset.tab)));
function plot(signal){
 const training=signal==='training';const series=training?data.training_series:data.trace;
 let lines=training?[{name:'0.006 rad',key:'large',color:'#a47744'},{name:'0.00006 rad',key:'micro',color:'#136f63'}]:[{name:signal==='turn'?'实际转角':'对向夹持',key:signal,color:'#136f63'}];
 const width=Math.max(300,$('#chart').clientWidth),height=235,left=50,right=16,top=16,bottom=32;const finite=[];
 lines.forEach(line=>series.forEach(p=>{if(Number.isFinite(p[line.key]))finite.push(p[line.key]);}));
 const xmin=training?0:0,xmax=training?32768:42.28;let ymin=signal==='pinch'?0:Math.min(0,...finite),ymax=signal==='pinch'?1:Math.max(...finite);if(ymin===ymax)ymax=ymin+1;
 const X=x=>left+(x-xmin)/(xmax-xmin)*(width-left-right),Y=y=>height-bottom-(y-ymin)/(ymax-ymin)*(height-top-bottom);
 let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${signal==='turn'?'实际转角':signal==='pinch'?'夹持状态':'训练检查点回报'}"><title>来自实际实验日志的数据</title>`;
 for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;svg+=`<line x1="${left}" y1="${Y(y)}" x2="${width-right}" y2="${Y(y)}" stroke="#dce4d7"/><text x="${left-10}" y="${Y(y)+4}" text-anchor="end" font-size="11" fill="#74816f">${signal==='pinch'?y.toFixed(2):Math.round(y)}</text>`;}
 for(let i=0;i<=4;i++){const x=xmax*i/4;svg+=`<text x="${X(x)}" y="${height-8}" text-anchor="middle" font-size="11" fill="#74816f">${training?Math.round(x).toLocaleString():x.toFixed(1)+'s'}</text>`;}
 lines.forEach((line,index)=>{const points=series.filter(p=>Number.isFinite(p[line.key])).map(p=>`${X(p.x)},${Y(p[line.key])}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${line.color}" stroke-width="2.5" vector-effect="non-scaling-stroke"/><text x="${width-205}" y="${24+index*18}" fill="${line.color}" font-size="11">${line.name}</text>`;});
 $('#chart').innerHTML=svg+'</svg>';$('#chart-heading').textContent=training?'实际PPO检查点评估回报':signal==='turn'?'实际转角随时间变化':'对向夹持状态随时间变化';$('#chart-caption').textContent=training?'同一参考来源、同种子、同预算。回报并非任务成功率，必须结合终态、掉落和穿透门槛。':'来自连续MuJoCo回放；夹持切换不是有效步态的充分证据。完整日志可从上方链接查看。';
}
$('#chart-signal').addEventListener('change',e=>plot(e.target.value));
window.addEventListener('resize',()=>plot($('#chart-signal').value));
$('#copy-code').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#commands').textContent);$('#copy-code').textContent='已复制';}catch{const range=document.createRange();range.selectNodeContents($('#commands'));window.getSelection().removeAllRanges();window.getSelection().addRange(range);$('#copy-code').textContent='已选中，请复制';}setTimeout(()=>$('#copy-code').textContent='复制命令',2200);});
resultTab('mujoco');plot('turn');
