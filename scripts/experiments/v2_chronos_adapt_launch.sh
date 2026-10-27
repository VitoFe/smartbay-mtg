#!/bin/bash
# Addendum 41 on Trinity: the zero-shot job and the four fine-tuning jobs on GPU 1, three threads
# each. Pass --quick for the smoke run. Run detached from the snapshot directory:
#   setsid nohup bash v2_chronos_adapt_launch.sh [--quick] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
Q="$1"
tag=${Q:+smoke_}
echo "start ${Q} $(date -u +%FT%TZ)" >> logs/campaign.status
pids=()
$P -u v2_chronos_adapt.py --zero-shot $Q > logs/${tag}zero_shot.log 2>&1 &
pids+=($!)
for v in temperature oxygen; do
  for m in full lora; do
    sleep 10
    $P -u v2_chronos_adapt.py --variable $v --method $m $Q > logs/${tag}train_${v}_${m}.log 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "done ${Q} failed_jobs=$fail $(date -u +%FT%TZ)" >> logs/campaign.status
