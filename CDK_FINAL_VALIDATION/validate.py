import argparse, os, json, time, pathlib, sys, itertools
import numpy as np, pandas as pd, joblib, sklearn, rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--target',choices=['CDK2','CDK4','CDK6'],required=True); p.add_argument('--model',choices=['SVR','ET','RF','XGB'],required=True); p.add_argument('--threads',type=int,default=12); p.add_argument('--stage',choices=['core','loo','all'],default='all'); a=p.parse_args()
root=pathlib.Path(a.root); out=root/'CDK_FINAL_VALIDATION'/'outputs'/a.target/a.model; out.mkdir(parents=True,exist_ok=True)
t=a.target; alg=a.model; seed=2024

def save(name,obj):
 f=out/name; tmp=f.with_name(f.name+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)); tmp.replace(f)
def load(name,default=None):
 f=out/name; return json.loads(f.read_text()) if f.exists() else default
def csv(name,d):
 f=out/name; tmp=f.with_name(f.name+'.tmp'); d.to_csv(tmp,index=False); tmp.replace(f)
def metrics(y,p):
 return dict(R2=float(r2_score(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)))
def make(par):
 if alg=='SVR': return SVR(kernel='rbf',cache_size=1024,**par)
 if alg=='ET': return ExtraTreesRegressor(random_state=seed,n_jobs=a.threads,**par)
 if alg=='RF': return RandomForestRegressor(random_state=seed,n_jobs=a.threads,**par)
 import xgboost as xgb
 return xgb.XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=seed,n_jobs=a.threads,**par)

grids={'SVR':dict(C=[1,10,50,100],gamma=[.001,.005,.01,.02,.05],epsilon=[.01,.05,.1,.2]),'ET':dict(n_estimators=[100,300,500],max_features=[.1,.3,'sqrt'],min_samples_leaf=[1,2,4]),'RF':dict(n_estimators=[100,300,500],max_features=[.1,.3,'sqrt'],min_samples_leaf=[1,2,4]),'XGB':dict(n_estimators=[100,300,500],max_depth=[3,5,7],learning_rate=[.01,.05,.1],subsample=[.8,1.0])}
print('START',t,alg, a.stage,flush=True)
d=pd.read_csv(root/'files'/f'{t}_provisional_model_ready.csv'); old=pd.read_csv(root/'CDK_pilot_results'/f'{t}_test_predictions.csv'); assert d.SMILES.is_unique and old.SMILES.is_unique
tr=d[~d.SMILES.isin(old.SMILES)].reset_index(drop=True); te=d.set_index('SMILES').loc[old.SMILES].reset_index(); assert np.allclose(te.pIC50,old.observed); assert not set(tr.scaffold.fillna('')).intersection(te.scaffold.fillna(''))
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def features(s):
 m=[Chem.MolFromSmiles(v) for v in s]; assert all(v is not None for v in m); return np.stack([gen.GetFingerprintAsNumPy(v) for v in m]),[gen.GetFingerprint(v) for v in m]
X,fp=features(tr.SMILES); Z,fz=features(te.SMILES); y=tr.pIC50.to_numpy(); z=te.pIC50.to_numpy(); groups=tr.scaffold.fillna('').astype(str).to_numpy(); n=len(y)
folds=list(GroupKFold(4).split(X,y,groups)); assert all(not set(groups[i])&set(groups[j]) for i,j in folds)
import hashlib
signature=dict(target=t,algorithm=alg,n_train=n,n_test=len(te),seed=seed,grid=grids[alg],python=sys.version,sklearn=sklearn.__version__,rdkit=rdkit.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__,data_sha256=hashlib.sha256((root/'files'/f'{t}_provisional_model_ready.csv').read_bytes()).hexdigest(),holdout_sha256=hashlib.sha256((root/'CDK_pilot_results'/f'{t}_test_predictions.csv').read_bytes()).hexdigest())
if alg=='XGB':
 import xgboost
 signature['xgboost']=xgboost.__version__
prior=load('run_config.json'); assert prior is None or prior==signature,'Configuration changed; use new output directory'; save('run_config.json',signature)
def cv(par, labels):
 pred=np.empty(n); scores=[]
 for i,j in folds:
  model=make(par); model.fit(X[i],labels[i]); pred[j]=model.predict(X[j]); scores.append(metrics(labels[j],pred[j])['RMSE'])
 return float(np.mean(scores)),pred
if a.stage in ('core','all'):
 grid=list(ParameterGrid(grids[alg])); results=load('grid_progress.json',[])
 for k in range(len(results),len(grid)):
  score,_=cv(grid[k],y); results.append(dict(params=grid[k],mean_fold_RMSE=score)); save('grid_progress.json',results); print('GRID',k+1,len(grid),score,flush=True)
 best=min(results,key=lambda r:r['mean_fold_RMSE']); save('best_params.json',best); csv('grid_results.csv',pd.DataFrame([dict(**r['params'],mean_fold_RMSE=r['mean_fold_RMSE']) for r in results]))
 par=best['params']; m=make(par); m.fit(X,y); pred=m.predict(Z); q=te[['SMILES','scaffold','pIC50']].copy(); q['predicted']=pred; q['similarity']=[max(DataStructs.BulkTanimotoSimilarity(f,fp)) for f in fz]; csv('holdout_predictions.csv',q); save('holdout_metrics.json',metrics(z,pred)); tmp=out/'model.tmp'; joblib.dump(m,tmp); tmp.replace(out/'model.joblib'); print('HOLDOUT',metrics(z,pred),flush=True)
 ad=[]
 for th in [0,.3,.4,.5,.6]:
  mask=q.similarity.to_numpy()>=th; row=dict(threshold=th,n=int(mask.sum()),coverage=float(mask.mean()));
  if mask.sum()>1: row.update(metrics(z[mask],pred[mask]));
  ad.append(row)
 csv('ad_metrics.csv',pd.DataFrame(ad))
 # Training-only repeated scaffold splits with fixed selected parameters; not nested.
 reps=load('repeated_progress.json',[])
 splits=list(GroupShuffleSplit(n_splits=20,test_size=.2,random_state=seed).split(X,y,groups))
 for k in range(len(reps),20):
  i,j=splits[k]; mm=make(par); mm.fit(X[i],y[i]); pr=mm.predict(X[j]); reps.append(dict(repeat=k+1,n_train=len(i),n_test=len(j),**metrics(y[j],pr))); save('repeated_progress.json',reps); print('REPEAT',k+1,flush=True)
 csv('repeated_metrics.csv',pd.DataFrame(reps))
 # Fixed-hyperparameter response permutation control; identical grouped folds.
 score,obs=cv(par,y); save('observed_cv.json',dict(mean_fold_RMSE=score,**metrics(y,obs))); rr=load('yrand_progress.json',[])
 for k in range(len(rr),30):
  yp=np.random.default_rng(seed+k+1).permutation(y); sc,pr=cv(par,yp); rr.append(dict(permutation=k+1,mean_fold_RMSE=sc,**metrics(yp,pr))); save('yrand_progress.json',rr); print('YRAND',k+1,flush=True)
 csv('yrand_metrics.csv',pd.DataFrame(rr)); save('yrand_summary.json',dict(observed_mean_fold_RMSE=score,empirical_p=(1+sum(r['mean_fold_RMSE']<=score for r in rr))/(1+len(rr)),n_permutations=len(rr))); save('core_complete.json',{'complete':True}); print('CORE COMPLETE',flush=True)
if a.stage in ('loo','all'):
 best=load('best_params.json'); assert best is not None,'Run core stage first'; par=best['params']; vals=load('loo_progress.json',[])
 for k in range(len(vals),n):
  mask=np.arange(n)!=k; mm=make(par); mm.fit(X[mask],y[mask]); vals.append(float(mm.predict(X[k:k+1])[0])); save('loo_progress.json',vals)
  if (k+1)%25==0 or k+1==n: print('LOO',k+1,n,flush=True)
 q=tr[['SMILES','scaffold','pIC50']].copy(); q['predicted']=vals; csv('loo_predictions.csv',q); m=metrics(y,vals); m['Q2_LOO']=m['R2']; save('loo_metrics.json',m); print('LOO COMPLETE',m,flush=True)
