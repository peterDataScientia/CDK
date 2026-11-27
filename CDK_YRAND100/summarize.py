import pathlib,json,pandas as pd
root=pathlib.Path(__file__).parent
rows=[]
for p in sorted((root/'outputs').glob('*/*')):
 if not (p/'complete.json').exists():continue
 s=json.loads((p/'yrand_summary.json').read_text())
 d=pd.read_csv(p/'yrand_metrics.csv')
 rows.append(dict(target=p.parent.name,model=p.name,**s,random_RMSE_mean=d.mean_fold_RMSE.mean(),random_RMSE_SD=d.mean_fold_RMSE.std(),random_R2_mean=d.R2.mean(),random_R2_SD=d.R2.std()))
r=pd.DataFrame(rows);r.to_csv(root/'summary.csv',index=False);print(r.round(6).to_string(index=False))
