import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage18'
def test_initialization_and_gradient_audit():
 init=json.loads((OUT/'initializations/rglcd_matched_e0_seed42.audit.json').read_text());assert init['shared_identical'] and init['gamma']==0 and init['phi_nonzero']
 grad=json.loads((OUT/'rglcd_gradient_audit.json').read_text());assert grad['initial']=={'box_max_abs':0.0,'score_max_abs':0.0,'decoded_max_abs':0.0};assert grad['steps'][0]['gamma_grad']!=0;assert grad['steps'][0]['phi_grad']==0;assert grad['steps'][1]['phi_grad']>0;assert all(x['finite'] for x in grad['steps'])
def test_preholdout_stop_is_explicit():
 f=json.loads((OUT/'RGLCD_EXPERIMENT_FREEZE.json').read_text());assert f['status']=='STOPPED_BEFORE_FORMAL_CV';assert f['holdout_metrics_read'] is False and f['formal_runs_started'] is False
 s=json.loads((OUT/'rglcd_mechanism_summary.json').read_text());assert s['scope']=='fold0 train only';assert s['gt']['d_abs']>s['background']['d_abs'];assert s['gt']['reliability']>.98 and s['background']['reliability']>.98
