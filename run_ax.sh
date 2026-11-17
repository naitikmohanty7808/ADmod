#!/usr/bin/env bash
# IEEE Access revision training queue. Resumable: finished runs are skipped and an interrupted run
# continues from its last epoch (runs_*/<variant>_s<seed>/last.pt). Re-run this script to resume.
set -u
cd "$(dirname "$0")"
mkdir -p runs_ax/logs runs_sens/logs
EPOCHS=100

python3 -m admod.dataset --sensitivity >> runs_sens/logs/generate.log 2>&1 || exit 1

for seed in 0 1 2; do
  for v in ax_full ax_plain ax_no_gap ax_no_global ax_no_region ax_one_scan ax_direct; do
    [ -f "runs_ax/${v}_s${seed}/train_log.json" ] && continue
    python3 -m admod.train --variant "$v" --seeds "$seed" --epochs $EPOCHS --data data --runs runs_ax \
      --bf16 --threads 2 >> "runs_ax/logs/${v}_s${seed}.log" 2>&1 || exit 1
    [ -x ./backup.sh ] && ./backup.sh >> runs_ax/logs/backup.log 2>&1
  done
done

for s in noise_0.5 noise_1.0 noise_1.5 noise_2.0 misreg_0.0 misreg_0.1 misreg_0.2 misreg_0.3 \
         rate_sd_low rate_sd_high susc_flat susc_steep; do
  [ -f "runs_sens/${s}/ax_full_s0/train_log.json" ] && continue
  python3 -m admod.train --variant ax_full --seeds 0 --epochs $EPOCHS --data "data/sens/${s}" \
    --runs "runs_sens/${s}" --bf16 --threads 2 >> "runs_sens/logs/${s}.log" 2>&1 || exit 1
  [ -x ./backup.sh ] && ./backup.sh >> runs_ax/logs/backup.log 2>&1
done
echo "training queue finished $(date -Is)" >> runs_ax/logs/queue.log

# Post hoc: same-session bfloat16 versus float32 parity check (ADmod, seed 1, 60 epochs, same data)
for prec in bf16 fp32; do
  [ -f "runs_parity/${prec}/ax_full_s1/train_log.json" ] && continue
  flag=""; [ "$prec" = bf16 ] && flag="--bf16"
  python3 -m admod.train --variant ax_full --seeds 1 --epochs 60 --data data --runs "runs_parity/${prec}" \
    $flag --threads 2 >> "runs_ax/logs/parity_${prec}.log" 2>&1 || exit 1
  [ -x ./backup.sh ] && ./backup.sh >> runs_ax/logs/backup.log 2>&1
done
echo "parity finished $(date -Is)" >> runs_ax/logs/queue.log
