#!/bin/bash
# Addendum 43 on Trinity: the four refits in parallel on GPU 1 (about 20 GB at peak), three threads
# each, then finalize once all have exited. Pass --quick for the smoke run.
#   setsid nohup bash v2_chronos_refit_launch.sh [--quick] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
Q="$1"
tag=${Q:+smoke_}
echo "start ${Q} $(date -u +%FT%TZ)" >> logs/campaign.status
pids=()
for job in timeval_full_temperature_raw timeval_lora_temperature_raw timeval_full_oxygen_aou timeval_lora_oxygen_aou; do
  $P -u v2_chronos_refit.py --job $job $Q > logs/${tag}${job}.log 2>&1 &
  pids+=($!)
  sleep 15
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "jobs done ${Q} failed=$fail $(date -u +%FT%TZ)" >> logs/campaign.status
$P -u v2_chronos_refit.py --finalize $Q > logs/${tag}finalize.log 2>&1
echo "refit finalize ${Q} $? $(date -u +%FT%TZ)" >> logs/campaign.status
