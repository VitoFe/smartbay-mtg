#!/bin/bash
# Addendum 40 on Trinity: three study workers per task on GPU 1, three threads each, then the
# finalize step once all six have exited. Run detached from the snapshot directory:
#   setsid nohup bash v2_chronos_lora_launch.sh > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
echo "start $(date -u +%FT%TZ)" >> logs/campaign.status
pids=()
for v in temperature oxygen; do
  for w in 0 1 2; do
    $P -u v2_chronos_lora.py --only $v > logs/worker_${v}_${w}.log 2>&1 &
    pids+=($!)
    sleep 20
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "workers done fail=$fail $(date -u +%FT%TZ)" >> logs/campaign.status
$P -u v2_chronos_lora.py --finalize > logs/finalize.log 2>&1
echo "finalize $? $(date -u +%FT%TZ)" >> logs/campaign.status
