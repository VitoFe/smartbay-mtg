#!/bin/bash
# Addendum 42 resume after the shared disk filled (2026-09-24 15:00 UTC): the two temperature
# studies continue to 100 completed trials (4 workers for full fine-tuning, 3 for adapters, sized to
# about 36.5 GB of GPU 1 at peak), and finalize runs only after every worker has exited.
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
echo "resume $(date -u +%FT%TZ)" >> logs/campaign.status
pids=()
for w in 0 1 2 3; do
  $P -u v2_chronos_timeval.py --only temperature --method full > logs/resume_temperature_full_$w.log 2>&1 &
  pids+=($!); sleep 20
done
for w in 0 1 2; do
  $P -u v2_chronos_timeval.py --only temperature --method lora > logs/resume_temperature_lora_$w.log 2>&1 &
  pids+=($!); sleep 20
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "resume workers done failed=$fail $(date -u +%FT%TZ)" >> logs/campaign.status
$P -u v2_chronos_timeval.py --finalize > logs/finalize_resumed.log 2>&1
echo "resumed finalize: $? $(date -u +%FT%TZ)" >> logs/campaign.status
