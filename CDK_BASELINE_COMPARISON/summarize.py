import pathlib,json,pandas as pd
root=pathlib.Path(__file__).parent
rows=[]
for p in sorted((root/'outputs').glob('*/*')):
 if not (p/'complete.json').exists():continue
 q=pd.read_csv(p/'paired_comparison.csv')
 r=dict(target=p.parent.name,model=p.name,source=json.loads((p/'complete.json').read_text())['source'],repeats=len(q))
 for k in ['R2','RMSE','MAE']:
  for s in ['baseline','optimized']:
   r[k+'_'+s]=q[k+'_'+s].mean()
   r[k+'_'+s+'_SD']=q[k+'_'+s].std()
  r[k+'_improvement']=q[k+'_improvement'].mean()
  r[k+'_wins']=int((q[k+'_improvement']>0).sum())
 rows.append(r)
d=pd.DataFrame(rows);d.to_csv(root/'baseline_summary.csv',index=False)
print(d.round(4).to_string(index=False))
