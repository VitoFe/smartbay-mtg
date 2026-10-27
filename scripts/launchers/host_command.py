"""Retrying connection to the compute host through the repo's helper; never prints settings.

usage: python trinity.py run "<cmd>" ["<cmd>" ...]
       python trinity.py get <remote> <local>
       python trinity.py put <local> <remote>
<base> in any argument is replaced by the remote base directory, and replaced back in output.
"""
import logging
import pathlib
import sys
import time

logging.disable(logging.CRITICAL)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
REPO = pathlib.Path(r'C:\Users\vferri\git\smartbay-do')
sys.path.insert(0, str(REPO / 'scripts'))
from scientific_campaign import connect, execute  # noqa: E402


def session(tries=6):
    for i in range(tries):
        try:
            return connect()
        except Exception as e:  # banner or session errors through the tunnel
            print(f'[connect attempt {i + 1} failed: {type(e).__name__}]', flush=True)
            time.sleep(4 + 3 * i)
    sys.exit('unreachable')


def main():
    op, args = sys.argv[1], sys.argv[2:]
    c, base = session()
    sub = lambda s: s.replace('<base>', base)
    try:
        if op == 'run':
            for cmd in args:
                code, out = execute(c, sub(cmd))
                print(f'$ {cmd[:110]}  [exit {code}]')
                print(out.rstrip().replace(base, '<base>')[:6000])
                print()
        elif op == 'get':
            c.open_sftp().get(sub(args[0]), args[1]); print('got', args[0])
        elif op == 'put':
            c.open_sftp().put(args[0], sub(args[1])); print('put', args[1])
    finally:
        c.close()


if __name__ == '__main__':
    main()
