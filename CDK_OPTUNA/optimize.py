import argparse, os, json, time, pathlib, sys, itertools
import numpy as np, pandas as pd, joblib, sklearn, rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--target',choices=['CDK2','CDK4','CDK6'],required=True); p.add_argument('--model',choices=['SVR','ET','RF','XGB'],required=True); p.add_argument('--threads',type=int,default=12); p.add_argument('--trials',type=int,default=100); a=p.parse_args()
root=pathlib.Path(a.root); out=root/'CDK_OPTUNA'/'outputs'/a.target/a.model; out.mkdir(parents=True,exist_ok=True)
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
print('START',t,alg, 'TPE',flush=True)
d=pd.read_csv(root/'files'/f'{t}_provisional_model_ready.csv'); old=pd.read_csv(root/'CDK_pilot_results'/f'{t}_test_predictions.csv'); assert d.SMILES.is_unique and old.SMILES.is_unique
tr=d[~d.SMILES.isin(old.SMILES)].reset_index(drop=True); te=d.set_index('SMILES').loc[old.SMILES].reset_index(); assert np.allclose(te.pIC50,old.observed); assert not set(tr.scaffold.fillna('')).intersection(te.scaffold.fillna(''))
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def features(s):
 m=[Chem.MolFromSmiles(v) for v in s]; assert all(v is not None for v in m); return np.stack([gen.GetFingerprintAsNumPy(v) for v in m]),[gen.GetFingerprint(v) for v in m]
X,fp=features(tr.SMILES); y=tr.pIC50.to_numpy(); groups=tr.scaffold.fillna('').astype(str).to_numpy(); n=len(y)
folds=list(GroupKFold(4).split(X,y,groups)); assert all(not set(groups[i])&set(groups[j]) for i,j in folds)
import hashlib
signature=dict(target=t,algorithm=alg,n_train=n,n_test=len(te),seed=seed,grid=grids[alg],python=sys.version,sklearn=sklearn.__version__,rdkit=rdkit.__version__,numpy=np.__version__,pandas=pd.__version__,joblib=joblib.__version__,data_sha256=hashlib.sha256((root/'files'/f'{t}_provisional_model_ready.csv').read_bytes()).hexdigest(),holdout_sha256=hashlib.sha256((root/'CDK_pilot_results'/f'{t}_test_predictions.csv').read_bytes()).hexdigest())
if alg=='XGB':
 import xgboost
 signature['xgboost']=xgboost.__version__
import optuna
signature['optuna']=optuna.__version__
signature['search_space']='TPE-v1'
signature['threads']=a.threads
prior=load('run_config.json'); assert prior is None or prior==signature,'Configuration changed; use new output directory'; save('run_config.json',signature)
def cv(par, labels):
 pred=np.empty(n); scores=[]
 for i,j in folds:
  model=make(par); model.fit(X[i],labels[i]); pred[j]=model.predict(X[j]); scores.append(metrics(labels[j],pred[j])['RMSE'])
 return float(np.mean(scores)),pred

# Same training rows, fingerprints, grouped folds and scoring as grid baseline.
# Never evaluate the holdout during TPE.
from optuna.trial import TrialState
baseline_path=root/'CDK_FINAL_VALIDATION'/'outputs'/t/alg/'best_params.json'
baseline=json.loads(baseline_path.read_text())
save('grid_baseline.json',baseline)

def objective(trial):
 if alg=='SVR':
  par=dict(C=trial.suggest_float('C',.01,1000,log=True),gamma=trial.suggest_float('gamma',1e-5,1,log=True),epsilon=trial.suggest_float('epsilon',.001,.5,log=True))
 elif alg in ('RF','ET'):
  depth=trial.suggest_int('depth_code',0,48)
  par=dict(n_estimators=trial.suggest_int('n_estimators',100,1000,step=50),max_features=trial.suggest_float('max_features',.02,1),max_depth=None if depth==0 else depth+2,min_samples_split=trial.suggest_int('min_samples_split',2,20),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,20))
 else:
  par=dict(n_estimators=trial.suggest_int('n_estimators',100,1000,step=50),max_depth=trial.suggest_int('max_depth',2,10),learning_rate=trial.suggest_float('learning_rate',.001,.3,log=True),subsample=trial.suggest_float('subsample',.5,1),colsample_bytree=trial.suggest_float('colsample_bytree',.1,1),reg_alpha=trial.suggest_float('reg_alpha',1e-6,10,log=True),reg_lambda=trial.suggest_float('reg_lambda',1e-6,100,log=True),min_child_weight=trial.suggest_int('min_child_weight',1,20))
 trial.set_user_attr('model_params',par)
 score,_=cv(par,y)
 return score

# One writer per database; no parallel trials or duplicate PBS jobs.
study=optuna.create_study(study_name=t+'_'+alg,storage='sqlite:///'+str(out/'study.sqlite3'),direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=20),load_if_exists=True)
# Interrupted trials are not counted toward the completed-trial budget.
for trial in study.get_trials(deepcopy=False,states=(TrialState.RUNNING,)):
 study.tell(trial.number,state=TrialState.FAIL)
def snapshot():
 done=study.get_trials(deepcopy=False,states=(TrialState.COMPLETE,))
 if not done:return
 b=study.best_trial
 opt=dict(params=b.user_attrs['model_params'],mean_fold_RMSE=b.value,trial=b.number,n_complete=len(done))
 save('tpe_best.json',opt)
 selected=dict(baseline,source='Grid') if baseline['mean_fold_RMSE']<=b.value else dict(opt,source='TPE')
 save('selected_candidate.json',selected)
 csv('trials.csv',study.trials_dataframe())
 print('TPE',len(done),'/',a.trials,'BEST',b.value,'GRID',baseline['mean_fold_RMSE'],flush=True)
snapshot()
remaining=max(0,a.trials-len(study.get_trials(deepcopy=False,states=(TrialState.COMPLETE,))))
study.optimize(objective,n_trials=remaining,n_jobs=1,callbacks=[lambda s,tr:snapshot()])
snapshot()
assert len(study.get_trials(deepcopy=False,states=(TrialState.COMPLETE,)))>=a.trials
save('tpe_complete.json',{'complete':True,'trials':a.trials})
