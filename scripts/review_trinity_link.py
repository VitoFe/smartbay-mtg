"""Direct link to the compute host for the standalone experiment sweeps.

Mirrors scripts/scientific_campaign.py:connect(): the host, port, login and remote directory come
from the environment or the uncommitted .env file (src/settings.py). The explicit `host` and `port`
arguments exist for a tunnel reached at another address; nothing here prints a setting.
"""
import shlex
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.settings import ssh_settings  # noqa: E402


def connect(host=None, port=None):
    """Return (client, base)."""
    cfg = ssh_settings()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host or cfg["host"],
        int(port or cfg["port"]),
        cfg["user"],
        password=cfg["password"],
        key_filename=cfg["key_filename"],
        timeout=25, banner_timeout=30, auth_timeout=30,
    )
    client.get_transport().set_keepalive(15)
    return client, cfg["remote_dir"]


def execute(client, command, timeout=120):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if err:
        out += "\n" + err
    return code, out


def run(command, timeout=120):
    client, base = connect()
    try:
        return execute(client, f"cd {shlex.quote(base)} && {command}", timeout=timeout)
    finally:
        client.close()


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) or "echo linked; hostname; whoami"
    code, out = run(cmd)
    print(f"exit={code}\n{out}")
