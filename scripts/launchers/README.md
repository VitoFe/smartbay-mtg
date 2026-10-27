# Launchers of addendum 82 (and two addendum-81 worker groups)

Parts A, B, E and G of addendum 82, and part of addendum 81's amendment-3 queue, were started on the
compute host by scripts kept outside the repository at the time. The files here are those scripts,
copied byte for byte: they are a record of what ran, not edited since. They read the connection
settings through `scripts/scientific_campaign.connect()` and print none of them. Both Python files that
run locally hard-code the checkout they ran from (`REPO`) and, for `add82_BE_launch.py`, keep their
packs under their own folder (`runs/`) and read the snapshot file lists (`trace_*.json`) beside
themselves; to reuse them, change those lines in a copy, not here. No run record lists a digest for
these files, so the digests below are the ones computed when they were copied.

| File | sha256 |
|---|---|
| `add82_BE_launch.py` | `da7bbd7469115a1b04ca018a0b06340f22db72d0532e6642f9303f135e100805` |
| `add82_E_requeue_gpu0.py` | `09ae7c64ae09dd1612885f03786e63d10cf357f599b68ece36f0d83dee0a0ba9` |
| `host_command.py` | `f16afed1710cf6c2794cdf7da682e1030dbaf23975982b0529dc2c91ff6692e5` |

## `add82_BE_launch.py`: parts B and E

`pack` builds a snapshot of the code and inputs with a sha256 manifest and one job script per study,
`upload` sends it, unpacks it on the host and checks every digest (writing `runtime.json`), `start` runs
the job scripts through a bounded `xargs` queue detached with `setsid -f`, `status` polls, and `fetch`
bundles the outputs on the host, checks the bundle's digest here and unpacks it under
`reports/bay_twin/runs/v2_addendum82/{B,E}/`, refusing to overwrite a different file. Runs it produced
(host folders `<base>/review_20260919/v2_add82_B/` and `.../v2_add82_E/`):

- `20261012T081329Z_add82B_smoke`: 5 quick jobs, never fetched.
- `20261012T081855Z_add82B`: 72 jobs, 8 at a time, CPU only; record in
  `reports/bay_twin/runs/v2_addendum82/B/run_20261012T081855Z_add82B/`.
- `20261012T082119Z_add82E_smoke`: 2 quick jobs, never fetched.
- `20261012T083217Z_add82Egates`: the 7 refits of the published selections, 3 at a time on GPU 1.
- `20261012T083248Z_add82E`: the reproduction study and 39 cell studies, 3 at a time on GPU 1 (see the
  next file for the second runner).

The file was edited once after the part-B runs were packed: the reproduction study moved from the gates
run to the study run (two lines in `pack`, in the branch part B never reaches). The version that packed
both part-B runs and the E smoke run hashes to
`248f7312bcdc3117f13cae06f5d5b98a34c462300f4f9f521740516c04beb223`; the file here is the later one,
which packed the two E runs and uploaded, started and fetched every run. Two E packs made before that
edit were never uploaded.

## `add82_E_requeue_gpu0.py`: part E's second runner

Run on the host inside `20261012T083248Z_add82E` once the GPU-1 queue was under way. It gives every job
not yet started a claim guard (`mkdir claims/<name>`), writes a GPU-0 copy of each such job script to
`jobs0/` and the reversed list to `queue0.txt`, so each job runs exactly once. The second runner was then
started inline, through `host_command.py`:
`setsid -f bash -c 'xargs -a queue0.txt -P 3 -I{} bash jobs0/{}.sh > queue0.log 2>&1; date -u > queue0.done'`.
It shared the 18 jobs not yet started (part E's run record gives the timing). The copy on the host has the
same sha256 as the file here.

## `host_command.py`: the inline launches

A small retrying wrapper around `scientific_campaign.connect()`: `run "<cmd>"`, `get <remote> <local>`,
`put <local> <remote>`, with `<base>` standing for the remote base directory. Parts A and G had no
launcher file: they were started by commands sent through it, with these steps.

- **Part A** (host folder `<base>/review_20260919/v2_add82_A/`). The snapshot (49 files) was packed by
  a one-off inline script; its manifest is `reports/bay_twin/runs/v2_addendum82/A/host_input_manifest.json`.
  It was uploaded as `snap.tar.gz` and unpacked into `snap/`. From `snap/`, with
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=V2_THREADS=2`, each detached with `setsid -f`:
  two `--quick` smoke jobs, then `python -u -m src.experiments.v2_rollout_a82_studies --variable V --arm A`
  for the weather tree of each variable on the CPU (`CUDA_VISIBLE_DEVICES` empty), one process each, and
  the four graph studies (temperature and oxygen, `conservative_exchange` and `learned_source`) one after
  another in a single chain on GPU 1. Logs are under `reports/bay_twin/runs/v2_addendum82/A/logs/host/`,
  studies under `.../A/studies/`.
- **Part G** (host folder `<base>/review_20260919/v2_add82_G/`). The inputs under
  `reports/bay_twin/runs/v2_addendum82/G/inputs/` and `scripts/experiments/v2_add82_g_run.py` were
  uploaded as one tar; the addendum-81 runner, its model layer, its input and output format and
  `weights.json` were copied from the addendum-81 run folder on the host (each equal to the repository's
  file). One detached chain on GPU 1 (`HF_HUB_OFFLINE=1`, 4 threads per library) ran
  `v2_add82_g_run.py check-f`, `run` and `determinism`; its logs and status are in
  `reports/bay_twin/runs/v2_addendum82/G/remote/`.
- **Addendum 81, amendment 3**. Besides the workers `scripts/v2_fm_zoo_remote.py a3-launch` started on
  GPU 1, five workers ran on GPU 0, started inline in the addendum-81 run folder:
  `a3g0w1-tfm3BCDE` to `a3g0w3-tfm3BCDE` (`--arms timesfm3_uni,timesfm3_cov --tasks BCDE --batch 1`),
  `a3g0w4-tfm3AF` (`--tasks AF --batch 1 --tfm3-flags make_positive=false`) and `a3g0w5-tfm25AF`
  (`--arms timesfm25_uni,timesfm25_cov --tasks AF --tfm25-config infer_is_positive=false`), each
  `v2_fm_zoo.py run --worker <name> --out preds ... --amendment
  config/bay_reserved_protocol_v2_addendum81_amendment3.json --amendment-sha256 1c73ba1b...`, plus the
  smoke workers `smokea3` and `smokea3b` on GPU 1. Their logs are in `reports/bay_twin/runs/v2_fm_zoo/run_logs/`.

Addendum 83 (every physical feature at every lead, 1 to 72 h) was launched from the repository itself;
its run record (`reports/bay_twin/runs/v2_addendum83/run_record.json`) and snapshot manifest record these
digests:

- `add83_launch.py` (sha256 `72ed18e1554333f5dbe0224b3f7b95136ff32ffda9d41a8df0ca704a67137010`): packs a
  hash-verified snapshot, uploads it, starts the shared-selection sweep on the compute host
  (run `20261012T201026Z_add83`, 720 studies in 632 jobs, CPU only) and fetches its outputs (`fetch <run>`).
- `add83_queue.py` (sha256 `61570f99fdb014f87596090d98782f5e9816b6d9c2c73c0e5561d892b2bc8674`): the
  host-side queue that keeps at most the declared number of two-thread jobs running and holds the load
  average under 100.
- `add83_perfold_host.py` (sha256 `912ce91bbe1c33845263d68a787fe14fdb8ca0560ee40a59c1ba31e7b109a1a9`): the
  per-fold studies of the ISPRA-based arms, run on the compute host where that family's published per-fold
  studies ran.
- `add83_perfold_local.py` (sha256 `ae7f68c368ed112a9f2e01f4acd23a4d3ff88568871f077250a5ea4649ffa29b`): the
  other per-fold studies, run locally like every other published per-fold study.

Not copied: the pollers that only read a queue's status while B and E ran and the script that fetched
part G's outputs (they launch nothing and name the host's full address), and the helper that traced the
B and E snapshot file lists (the resulting lists are in each run folder's `snapshot_manifest.json`).
