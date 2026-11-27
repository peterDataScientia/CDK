#!/bin/bash
set -euo pipefail
cd /mnt/lustre/users/apeter1/CDK_NP_SCREENING/CDK_BASELINE_AD
mkdir -p logs
for t in CDK2 CDK4 CDK6; do
 for m in SVR ET RF XGB; do
  [[ -f outputs/$t/$m/complete.json ]] && continue
  qsub -N "${t}_${m}_BAD" -o "$PWD/logs/${t}_${m}_$(date +%s).log" -v TARGET="$t",MODEL="$m" run.pbs
 done
done
