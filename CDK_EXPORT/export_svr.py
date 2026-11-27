import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import json
import gzip
import shutil

root=Path(__file__).resolve().parent.parent
out=root/'FINAL_SVR_SCREENING'
out.mkdir(exist_ok=True)
targets=['CDK2','CDK4','CDK6']
summary=[]
manifest=[]
for source in ['COCONUT','LOTUS']:
    path=root/f'CDK_{source}_full_SVR_ET.csv'
    d=pd.read_csv(path)
    assert d.SMILES.is_unique, source
    cols=[c for c in d if not c.endswith('_ET') and not c.startswith('ET_')]
    d=d[cols].copy()
    sim=[t+'_similarity' for t in targets]
    known=[t+'_known' for t in targets]
    assert all(pd.api.types.is_bool_dtype(d[c]) for c in known)
    assert np.isfinite(d[sim].to_numpy()).all()
    assert ((d[sim]>=0)&(d[sim]<=1)).all().all()
    assert np.isfinite(d[[t+'_SVR' for t in targets]].to_numpy()).all()
    assert np.allclose(d.SVR_dual,d[['CDK2_SVR','CDK4_SVR']].min(axis=1),atol=1e-6)
    assert np.allclose(d.SVR_margin,d.SVR_dual-d.CDK6_SVR,atol=1e-6)
    assert np.allclose(d.SVR_fold,10**d.SVR_margin,rtol=1e-5)
    d['AD_min']=d[sim].min(axis=1)
    unseen=d.loc[~d[known].any(axis=1)].copy()
    full=out/f'{source}_SVR_full_unseen.csv.gz'
    unseen.to_csv(full,index=False,compression='gzip')
    for ad in [0,.3,.4,.5,.6]:
        eligible=unseen[unseen.AD_min>=ad]
        dual=eligible[eligible.SVR_dual>=6].sort_values('SVR_fold',ascending=False)
        s2=dual[dual.SVR_fold>=2]
        s10=dual[dual.SVR_fold>=10]
        tag=f'AD{round(ad*100):03d}'
        for label,q in [('dual',dual),('selective2',s2),('selective10',s10)]:
            q.to_csv(out/f'{source}_{tag}_{label}.csv',index=False)
        summary.append(dict(Source=source,AD=ad,Total=len(d),Unseen=len(unseen),AD_eligible=len(eligible),Dual_hits=len(dual),Selectivity_2fold=len(s2),Selectivity_10fold=len(s10)))
    manifest.append(dict(source=source,input=path.name,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),records=len(d),unseen=len(unseen)))
    if full.stat().st_size>90*1024**2:
        with full.open('rb') as f:
            i=0
            while True:
                block=f.read(45*1024**2)
                if not block:break
                (out/(full.name+f'.part{i:03d}')).write_bytes(block)
                i+=1
        full.unlink()
    print(source,'EXPORT COMPLETE',flush=True)
r=pd.DataFrame(summary)
r.to_csv(out/'SVR_screening_summary.csv',index=False)
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'README.md').write_text('''# Baseline SVR natural-product screening
Existing predictions from screen_np_extended.py; no new fitting or prediction.
Original historical SVR models for CDK2, CDK4 and CDK6, not optimised SVR models.
Known compounds in any curated target dataset excluded. Joint AD = minimum of three target-specific nearest-training-neighbour similarities.
Dual hit: both predicted pIC50 >= 6. Predicted selectivity fold = 10**(min(CDK2,CDK4 pIC50)-CDK6 pIC50).
Candidates are computational hypotheses, not experimentally verified.
If gzip files are split, concatenate the numbered parts in order to reconstruct the original gzip stream.
''')
print(r.to_string(index=False))
print('EXPORT COMPLETE; NO NEW PREDICTIONS')
