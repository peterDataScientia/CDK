#!/bin/bash
set -euo pipefail
cd /mnt/lustre/users/apeter1/CDK_NP_SCREENING/CDK_OPTUNA
mkdir -p logs
for t in CDK2 CDK4 CDK6; do
 for m in SVR ET RF XGB; do
  if [[ -f outputs/$t/$m/tpe_complete.json ]]; then continue; fi
  [[ -f ../CDK_FINAL_VALIDATION/outputs/$t/$m/best_params.json ]] || { echo "Missing grid baseline $t $m"; continue; }
  qsub -N "${t}_${m}_TPE" -o "$PWD/logs/${t}_${m}_$(date +%s).log" -v TARGET="$t",MODEL="$m" run.pbs
 done
done
