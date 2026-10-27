"""Share the not-yet-started jobs of a running queue with a second runner on GPU 0.

Run in the run directory. Every job that has neither started (no log) nor finished (no status) gets a
claim guard (mkdir claims/<name>) in its GPU-1 script and a GPU-0 copy with the same guard; the GPU-0
runner takes them in reverse order, the running GPU-1 runner keeps its order, and each job runs once.
"""
import pathlib
import re

p = pathlib.Path.cwd()
names = [n for n in (p / 'queue.txt').read_text().split('\n') if n]
todo = [n for n in names if not (p / 'logs' / f'{n}.log').exists() and not (p / 'status' / f'{n}.exit').exists()]
(p / 'claims').mkdir(exist_ok=True)
(p / 'jobs0').mkdir(exist_ok=True)
for n in todo:
    src = (p / 'jobs' / f'{n}.sh').read_text()
    lines = src.split('\n')
    guard = f'mkdir claims/{n} 2>/dev/null || exit 0'
    if guard not in src:
        lines.insert(2, guard)          # after the shebang and the cd
    body = '\n'.join(lines)
    (p / 'jobs' / f'{n}.sh').write_text(body)
    (p / 'jobs0' / f'{n}.sh').write_text(re.sub(r"CUDA_VISIBLE_DEVICES='1'", "CUDA_VISIBLE_DEVICES='0'", body))
(p / 'queue0.txt').write_text('\n'.join(reversed(todo)) + '\n')
print('shared', len(todo), 'jobs:', ' '.join(todo))
