#!/bin/bash
# Addendum 46 on Trinity: the zero-shot look-back job on GPU 1, three threads. Pass --quick for the
# smoke run. Run detached from the snapshot directory:
#   setsid nohup bash v2_chronos_lookback_launch.sh [--quick] > /dev/null 2>&1 < /dev/null & disown
cd "$(dirname "$0")"
P=/home/vito/smartbay_do/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 V2_THREADS=3
Q="$1"
tag=${Q:+smoke_}
mkdir -p logs
echo "start ${Q} $(date -u +%FT%TZ)" >> logs/campaign.status
$P -u v2_chronos_lookback.py $Q > logs/${tag}lookback.log 2>&1
echo "done ${Q} exit=$? $(date -u +%FT%TZ)" >> logs/campaign.status
