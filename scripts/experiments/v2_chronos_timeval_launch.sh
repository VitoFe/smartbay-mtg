#!/bin/bash
# Addendum 42 on Trinity: two study workers per (variable, method), eight in all on GPU 1, three
# threads each, then the finalize step once all have exited. Pass --quick for the smoke run.
#   setsid nohup bash v2_chronos_timeval_launch.sh [--quick] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
Q="$1"
tag=${Q:+smoke_}
echo "start ${Q} $(date -u +%FT%TZ)" >> logs/campaign.status
pids=()
for v in temperature oxygen; do
  for m in full lora; do
    for w in 0 1; do
      $P -u v2_chronos_timeval.py --only $v --method $m $Q > logs/${tag}worker_${v}_${m}_${w}.log 2>&1 &
      pids+=($!)
      sleep 20
    done
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "workers done ${Q} failed=$fail $(date -u +%FT%TZ)" >> logs/campaign.status
$P -u v2_chronos_timeval.py --finalize $Q > logs/${tag}finalize.log 2>&1
echo "finalize ${Q} $? $(date -u +%FT%TZ)" >> logs/campaign.status
