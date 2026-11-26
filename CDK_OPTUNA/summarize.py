import pathlib,json,pandas as pd
root=pathlib.Path(__file__).parent
rows=[]
for t in ['CDK2','CDK4','CDK6']:
 for m in ['SVR','ET','RF','XGB']:
  p=root/'outputs'/t/m
  if not (p/'tpe_best.json').exists():continue
  b=json.loads((p/'grid_baseline.json').read_text())
  a=json.loads((p/'tpe_best.json').read_text())
  rows.append(dict(target=t,model=m,grid_RMSE=b['mean_fold_RMSE'],TPE_RMSE=a['mean_fold_RMSE'],improvement=b['mean_fold_RMSE']-a['mean_fold_RMSE'],trials=a['n_complete'],complete=(p/'tpe_complete.json').exists(),selected='TPE' if a['mean_fold_RMSE']<b['mean_fold_RMSE'] else 'Grid'))
d=pd.DataFrame(rows);d.to_csv(root/'optimization_summary.csv',index=False);print(d.to_string(index=False))
