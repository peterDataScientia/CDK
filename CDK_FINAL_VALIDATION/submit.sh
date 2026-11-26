#!/bin/bash
set -euo pipefail
cd /mnt/lustre/users/apeter1/CDK_NP_SCREENING/CDK_FINAL_VALIDATION
mkdir -p logs
stage=${1:-core}
[[ "$stage" == core || "$stage" == loo ]] || exit 2
for t in CDK2 CDK4 CDK6; do
 for m in SVR ET RF XGB; do
  if [[ "$stage" == core && -f outputs/$t/$m/core_complete.json ]]; then continue; fi
  if [[ "$stage" == loo && ! -f outputs/$t/$m/core_complete.json ]]; then echo "SKIP $t $m: core not complete"; continue; fi
  if [[ "$stage" == loo && -f outputs/$t/$m/loo_metrics.json ]]; then continue; fi
  qsub -N "${t}_${m}_${stage}" -o "$PWD/logs/${t}_${m}_${stage}_$(date +%s).log" -v TARGET="$t",MODEL="$m",STAGE="$stage" run.pbs
 done
done
