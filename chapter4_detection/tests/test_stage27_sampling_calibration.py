import math
from chapter4_detection.scripts.stage27_calibrate_scores import apply, fit_platt
from chapter4_detection.scripts.stage27_train_fspr import FixedBudgetNaturalSampler

def test_natural_sampler_fixed_budget_and_no_class_reweighting():
    s=FixedBudgetNaturalSampler(7,20,42); values=list(iter(s))
    assert len(values)==20 and set(values)==set(range(7))
    counts=[values.count(i) for i in range(7)]
    assert max(counts)-min(counts)<=1

def test_platt_scaling_orders_scores_and_is_finite():
    rows=[{'score':s,'label':int(s>.5)} for s in (.05,.1,.2,.8,.9,.95)]
    p=fit_platt(rows); values=[apply(s,p) for s in (.1,.5,.9)]
    assert p['status']=='fit' and values[0]<values[1]<values[2]
    assert all(math.isfinite(x) and 0<=x<=1 for x in values)
