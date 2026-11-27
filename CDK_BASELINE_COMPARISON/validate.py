import argparse, os, json, time, pathlib, sys, itertools
import numpy as np, pandas as pd, joblib, sklearn, rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--target',choices=['CDK2','CDK4','CDK6'],required=True); p.add_argument('--model',choices=['SVR','ET','RF','XGB'],required=True); p.add_argument('--threads',type=int,default=12); p.add_argument('--stage',choices=['core','loo','all'],default='all'); a=p.parse_args()
root=pathlib.Path(a.root); out=root/'CDK_BASELINE_COMPARISON'/'outputs'/a.target/a.model; out.mkdir(parents=True,exist_ok=True)
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

from sklearn.base import clone
locked=root/'CDK_LOCKED_VALIDATION'/'outputs'/t/alg
ref=json.loads((locked/'run_config.json').read_text())
for key in ['data_sha256','holdout_sha256','seed','n_train','n_test','sklearn','rdkit','numpy','pandas']:
 assert signature[key]==ref[key], 'Mismatch: '+key
if alg=='XGB': assert signature['xgboost']==ref['xgboost']
if alg in ('SVR','ET'):
 path=root/('CDK_pilot_results' if alg=='SVR' else 'CDK_boost/ET')/f'{t}_{alg}.joblib'
 base=clone(joblib.load(path))
 if alg=='ET': base.set_params(n_jobs=a.threads)
 if alg=='SVR': base.set_params(cache_size=1024)
 source='historical_saved_model'
elif alg=='RF':
 base=RandomForestRegressor(n_estimators=100,random_state=seed,n_jobs=a.threads)
 source='new_default_reference'
else:
 import xgboost as xgb
 base=xgb.XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=seed,n_jobs=a.threads)
 source='new_default_reference'
signature['baseline_params']=base.get_params()
signature['baseline_source']=source
# XGBoost may expose NaN as its missing-value marker; serialize metadata safely.
signature=json.loads(json.dumps(signature,default=str).replace(': NaN', ': "NaN"'))
prior=load('run_config.json'); assert prior is None or prior==signature
save('run_config.json',signature)
splits=list(GroupShuffleSplit(n_splits=20,test_size=.2,random_state=seed).split(X,y,groups))
opt=pd.read_csv(locked/'repeated_metrics.csv').sort_values('repeat')
assert list(opt.repeat)==list(range(1,21))
reps=load('repeated_progress.json',[])
for k in range(len(reps),20):
 i,j=splits[k]
 assert len(i)==opt.iloc[k].n_train and len(j)==opt.iloc[k].n_test
 assert not set(groups[i])&set(groups[j])
 mm=clone(base); mm.fit(X[i],y[i]); pr=mm.predict(X[j])
 reps.append(dict(repeat=k+1,n_train=len(i),n_test=len(j),**metrics(y[j],pr)))
 save('repeated_progress.json',reps); print('BASELINE REPEAT',k+1,flush=True)
b=pd.DataFrame(reps); csv('repeated_metrics.csv',b)
q=b.merge(opt,on=['repeat','n_train','n_test'],suffixes=('_baseline','_optimized'),validate='one_to_one')
for key in ['R2','RMSE','MAE']:
 q[key+'_improvement']=(q[key+'_optimized']-q[key+'_baseline'])*(1 if key=='R2' else -1)
csv('paired_comparison.csv',q)
mm=clone(base); mm.fit(X,y); save('holdout_metrics.json',metrics(z,mm.predict(Z)))
save('complete.json',{'complete':True,'source':source})
print('BASELINE COMPLETE',flush=True)
