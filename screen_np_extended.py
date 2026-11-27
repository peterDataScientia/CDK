import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import pandas as pd,numpy as np,joblib,zipfile,argparse
from pathlib import Path
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize as st
RDLogger.DisableLog('rdApp.*')
p=argparse.ArgumentParser();p.add_argument('--sample',type=int,default=0);p.add_argument('--source',choices=['LOTUS','COCONUT','ANPDB'],required=True);p.add_argument('--root',default='/mnt/data');a=p.parse_args();root=Path(a.root); gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048); targets=['CDK2','CDK4','CDK6']; rng=np.random.default_rng(2024)
if a.source=='LOTUS': d=pd.read_csv(root/'LOTUS_DB.smi',sep='\t',header=None,names=['smiles','identifier'],dtype=str)
elif a.source=='COCONUT': d=pd.read_csv(root/'coconut_csv-11-2026.zip',usecols=['identifier','canonical_smiles','name'],dtype=str).rename(columns={'canonical_smiles':'smiles'})
else: d=pd.read_csv(root/'CDK_ANPDB_screen.csv',usecols=['molecule_id','mol_name','SMILES']).rename(columns={'molecule_id':'identifier','SMILES':'smiles'})
print(a.source,'raw',len(d),flush=True)
if a.sample and len(d)>a.sample:d=d.sample(a.sample,random_state=2024)
def standard(s):
 try:
  m=Chem.MolFromSmiles(s)
  if a.source!='ANPDB':m=st.Uncharger().uncharge(st.FragmentParent(st.Cleanup(m)))
  return Chem.MolToSmiles(m,isomericSmiles=True) if any(x.GetAtomicNum()==6 for x in m.GetAtoms()) else None
 except Exception:return None
refs={};models={}
for t in targets:
 df=pd.read_csv(root/'files'/f'{t}_provisional_model_ready.csv'); test=pd.read_csv(root/'CDK_pilot_results'/f'{t}_test_predictions.csv');tr=df[~df.SMILES.isin(test.SMILES)];refs[t]=(set(df.SMILES),[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.SMILES]);models[t]=(joblib.load(root/'CDK_pilot_results'/f'{t}_SVR.joblib'),joblib.load(root/'CDK_boost'/'ET'/f'{t}_ET.joblib'))
out=root/f'CDK_{a.source}_{"sample" if a.sample else "full"}_SVR_ET.csv'; parts=root/f'{a.source}_{"sample"+str(a.sample) if a.sample else "full"}_parts';parts.mkdir(exist_ok=True); seen=set(); first=True
for start in range(0,len(d),512):
 part=parts/f'{start:09d}.csv.gz'
 if part.exists():
  old=pd.read_csv(part);seen.update(old.SMILES.dropna());continue
 q=d.iloc[start:start+512].copy();q['SMILES']=q.smiles.map(standard);q=q.dropna(subset=['SMILES']).drop_duplicates('SMILES');q=q[~q.SMILES.isin(seen)].copy();seen.update(q.SMILES)
 if q.empty:continue
 ms=[Chem.MolFromSmiles(s) for s in q.SMILES]
 valid=[i for i,m in enumerate(ms) if m is not None]
 q=q.iloc[valid].copy().reset_index(drop=True)
 ms=[ms[i] for i in valid]
 if not ms:
     continue
 fps=[gen.GetFingerprint(m) for m in ms]
 X=np.stack([gen.GetFingerprintAsNumPy(m) for m in ms]).astype('float32')
 for t in targets:
  svm,et=models[t];V=svm.support_vectors_.astype('float32');dist=np.maximum(0,X.sum(1)[:,None]+V.sum(1)[None,:]-2*X@V.T);q[t+'_SVR']=np.exp(-svm._gamma*dist)@svm.dual_coef_.ravel()+svm.intercept_[0];q[t+'_ET']=et.predict(X);q[t+'_similarity']=[max(DataStructs.BulkTanimotoSimilarity(f,refs[t][1])) for f in fps];q[t+'_known']=q.SMILES.isin(refs[t][0]);assert np.allclose(q[t+'_SVR'].iloc[:2],svm.predict(X[:2]),atol=1e-5)
 q['min_AD']=q[[t+'_similarity' for t in targets]].min(axis=1);q['unseen']=~q[[t+'_known' for t in targets]].any(axis=1)
 for model in ['SVR','ET']:
  q[model+'_dual']=q[[t+'_'+model for t in targets[:2]]].min(axis=1);q[model+'_margin']=q[model+'_dual']-q['CDK6_'+model];q[model+'_fold']=10**q[model+'_margin']
 tmp=parts/f'{start:09d}.tmp.gz';q.to_csv(tmp,index=False,compression='gzip');tmp.replace(part)
 print(a.source,start+len(q),'processed',flush=True)
for part in sorted(parts.glob('*.csv.gz')):
 pd.read_csv(part).to_csv(out,index=False,mode='w' if first else 'a',header=first);first=False
q=pd.read_csv(out);print('unique',len(q),'unseen',q.unseen.sum(),flush=True)
for m in ['SVR','ET']:
 for ad in [0,.3,.4,.5]:
  z=q[q.unseen&(q[m+'_dual']>=6)&(q.min_AD>=ad)];print(m,ad,len(z),int((z[m+'_fold']>=2).sum()),int((z[m+'_fold']>=10).sum()),flush=True)
