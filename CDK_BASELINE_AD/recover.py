"""Recover baseline AD results without fitting any models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

root=Path(__file__).resolve().parent.parent
out=root/'CDK_BASELINE_AD'
rows=[]; comparisons=[]; predictions=[]
def keyed(d):
    d=d.copy()
    d['AD_key']=np.rint(d.threshold*10).astype(int)
    assert np.allclose(d.threshold,d.AD_key/10,rtol=0,atol=1e-10)
    assert sorted(d.AD_key)==[0,3,4,5,6]
    return d.drop(columns='threshold')

for t in ['CDK2','CDK4','CDK6']:
    for m in ['SVR','ET','RF','XGB']:
        p=out/'outputs'/t/m
        o=root/'CDK_LOCKED_VALIDATION'/'outputs'/t/m
        b=root/'CDK_BASELINE_COMPARISON'/'outputs'/t/m
        baseline=pd.read_csv(p/'ad_metrics.csv')
        optimized=pd.read_csv(o/'ad_metrics.csv')
        h=pd.read_csv(p/'holdout_predictions.csv')
        ref=pd.read_csv(o/'holdout_predictions.csv')
        assert h.SMILES.is_unique and ref.SMILES.is_unique
        assert list(h.SMILES)==list(ref.SMILES)
        assert np.allclose(h.pIC50,ref.pIC50,rtol=0,atol=1e-10)
        assert np.allclose(h.similarity,ref.similarity,rtol=0,atol=1e-10)
        expected=json.loads((b/'holdout_metrics.json').read_text())
        for d,pred in [(baseline,h),(optimized,ref)]:
            for row in d.itertuples():
                mask=pred.similarity.to_numpy()>=row.threshold
                assert mask.sum()==row.n
                assert np.isclose(mask.mean(),row.coverage,rtol=0,atol=1e-10)
                y=pred.pIC50.to_numpy()[mask]; z=pred.predicted.to_numpy()[mask]
                actual={'R2':r2_score(y,z),'RMSE':np.sqrt(mean_squared_error(y,z)),'MAE':mean_absolute_error(y,z)}
                for key,v in actual.items():
                    assert np.isclose(v,getattr(row,key),rtol=0,atol=1e-6),(t,m,row.threshold,key)
                    if d is baseline and row.threshold==0:
                        assert np.isclose(v,expected[key],rtol=0,atol=1e-6)
        q=keyed(baseline).merge(keyed(optimized),on='AD_key',suffixes=('_baseline','_optimized'),validate='one_to_one')
        assert len(q)==5
        assert (q.n_baseline==q.n_optimized).all()
        assert np.allclose(q.coverage_baseline,q.coverage_optimized,rtol=0,atol=1e-10)
        q['threshold']=q.pop('AD_key')/10
        q['n']=q.pop('n_baseline'); q=q.drop(columns='n_optimized')
        q['coverage']=q.pop('coverage_baseline'); q=q.drop(columns='coverage_optimized')
        for key in ['RMSE','MAE']:
            q[key+'_improvement']=q[key+'_baseline']-q[key+'_optimized']
        q['R2_improvement']=q.R2_optimized-q.R2_baseline
        q.to_csv(p/'ad_comparison.csv',index=False)
        for d in [baseline,q,h]:
            d.insert(0,'Model',m);d.insert(0,'Target',t)
        rows.append(baseline);comparisons.append(q);predictions.append(h)
        print(t,m,'VERIFIED AND RECOVERED',flush=True)
r=pd.concat(rows,ignore_index=True); c=pd.concat(comparisons,ignore_index=True); h=pd.concat(predictions,ignore_index=True)
assert len(r)==len(c)==60 and len(h)==3428
for name,d in [('AD_summary.csv',r),('AD_comparison.csv',c),('ALL_baseline_holdout_predictions.csv',h)]:
    tmp=out/(name+'.tmp');d.to_csv(tmp,index=False);tmp.replace(out/name)
print(r.round(4).to_string(index=False))
print('RECOVERY SUCCESSFUL: 12 models; 60 paired AD records; 3428 predictions; zero refits')
