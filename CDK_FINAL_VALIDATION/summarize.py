import pathlib,json,pandas as pd,numpy as np
from sklearn.metrics import mean_squared_error
root=pathlib.Path(__file__).parent; out=root/'outputs'; rows=[]; joint=[]
for p in sorted(out.glob('*/*')):
 if not p.is_dir():continue
 for f in ['holdout_metrics.json','loo_metrics.json','observed_cv.json']:
  if (p/f).exists(): rows.append(dict(target=p.parent.name,model=p.name,stage=f[:-5],**json.loads((p/f).read_text())))
pd.DataFrame(rows).to_csv(root/'summary_metrics.csv',index=False)
for m in ['SVR','ET','RF','XGB']:
 paths=[out/t/m/'holdout_predictions.csv' for t in ['CDK2','CDK4','CDK6']]
 if not all(p.exists() for p in paths):continue
 ds=[pd.read_csv(p).set_index('SMILES') for p in paths]; ids=ds[0].index.intersection(ds[1].index).intersection(ds[2].index); q=pd.DataFrame(index=ids)
 for col in ['pIC50','predicted']:
  arr=[d.loc[ids,col].to_numpy() for d in ds]; q[col+'_dual']=np.minimum(arr[0],arr[1]); q[col+'_margin']=q[col+'_dual']-arr[2]
  q[col+'_hit']=(q[col+'_dual']>=6)&(q[col+'_margin']>=1)
 q.to_csv(root/f'joint_{m}.csv'); joint.append(dict(model=m,n=len(q),margin_RMSE=float(np.sqrt(mean_squared_error(q.pIC50_margin,q.predicted_margin))),observed_hits=int(q.pIC50_hit.sum()),predicted_hits=int(q.predicted_hit.sum())))
pd.DataFrame(joint).to_csv(root/'summary_joint.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(joint).to_string(index=False))
