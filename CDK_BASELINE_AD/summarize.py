import pathlib,json,pandas as pd
root=pathlib.Path(__file__).resolve().parent
rows=[]; comps=[]; preds=[]
for t in ['CDK2','CDK4','CDK6']:
 for m in ['SVR','ET','RF','XGB']:
  p=root/'outputs'/t/m
  if not (p/'complete.json').exists():
   print('MISSING',t,m); continue
  assert json.loads((p/'complete.json').read_text())['complete']
  for filename, dest in [('ad_metrics.csv',rows),('ad_comparison.csv',comps),('holdout_predictions.csv',preds)]:
   d=pd.read_csv(p/filename); d.insert(0,'Model',m); d.insert(0,'Target',t); dest.append(d)
assert len(rows)==12, f'Completed {len(rows)}/12'
r=pd.concat(rows,ignore_index=True); c=pd.concat(comps,ignore_index=True); h=pd.concat(preds,ignore_index=True)
assert len(r)==len(c)==60 and len(h)==3428
r.to_csv(root/'AD_summary.csv',index=False);c.to_csv(root/'AD_comparison.csv',index=False);h.to_csv(root/'ALL_baseline_holdout_predictions.csv',index=False)
print(r.round(4).to_string(index=False))
print('COMPLETE: 12/12; 60 paired AD records; 3428 predictions')
