"""Deployment settings and the data location, read from the environment.

A `.env` file at the repository root is read first, if present, without overriding variables that
are already set; `.env` is git-ignored and `.env.example` lists every key. Nothing secret has a
default here, and nothing in this module prints a value.

    SMARTBAY_DATA_DIR        where the data folder lives (default: <repo>/data)
    SMARTBAY_SSH_HOST        compute host reached over SSH
    SMARTBAY_SSH_PORT        its SSH port (default 22)
    SMARTBAY_SSH_USER        login
    SMARTBAY_SSH_PASSWORD    password, or leave empty and set
    SMARTBAY_SSH_KEY_FILE    a private key file (preferred)
    SMARTBAY_REMOTE_DIR      the project directory on the host
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / '.env'


def load_dotenv(path=ENV_FILE):
    """KEY=VALUE lines into os.environ; blank lines and # comments skipped; quotes stripped."""
    path = Path(path)
    if not path.is_file():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv()


def data_dir():
    """The data folder the code reads as <repo>/data (see scripts/data_setup.py)."""
    configured = os.environ.get('SMARTBAY_DATA_DIR', '').strip()
    return Path(configured).expanduser().resolve() if configured else ROOT / 'data'


def ssh_settings():
    """Connection settings for the compute host, or a clear error naming what is missing."""
    env = os.environ
    missing = [k for k in ('SMARTBAY_SSH_HOST', 'SMARTBAY_SSH_USER', 'SMARTBAY_REMOTE_DIR')
               if not env.get(k, '').strip()]
    if not (env.get('SMARTBAY_SSH_PASSWORD') or env.get('SMARTBAY_SSH_KEY_FILE')):
        missing.append('SMARTBAY_SSH_PASSWORD or SMARTBAY_SSH_KEY_FILE')
    if missing:
        raise RuntimeError('remote settings missing: ' + ', '.join(missing)
                           + f' (set them in the environment or in {ENV_FILE.name}; see .env.example)')
    key = env.get('SMARTBAY_SSH_KEY_FILE', '').strip()
    return {'host': env['SMARTBAY_SSH_HOST'].strip(),
            'port': int(env.get('SMARTBAY_SSH_PORT', '22') or 22),
            'user': env['SMARTBAY_SSH_USER'].strip(),
            'password': env.get('SMARTBAY_SSH_PASSWORD') or None,
            'key_filename': str(Path(key).expanduser()) if key else None,
            'remote_dir': env['SMARTBAY_REMOTE_DIR'].strip()}
