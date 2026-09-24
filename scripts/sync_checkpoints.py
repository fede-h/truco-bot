#!/usr/bin/env python3
"""Sync trained model checkpoints from remote Vast.ai instance to local storage."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def sync_models(
    host: str = "95.253.220.115",
    port: int = 62855,
    key_path: str = "~/.ssh/id_ed25519",
    remote_dir: str = "/workspace/truco-bot/models/",
    local_dir: str = "models/",
) -> list[str]:
    """Run rsync to pull models from remote instance."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    expanded_key = os.path.expanduser(key_path)

    cmd = [
        "rsync",
        "-avz",
        "--update",
        "--progress",
        "-e",
        f"ssh -p {port} -i {expanded_key} -o StrictHostKeyChecking=no",
        f"root@{host}:{remote_dir}",
        local_dir,
    ]

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Syncing from root@{host}:{port} -> {local_dir}...")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if res.returncode != 0:
        print(f"Sync error: {res.stderr.strip()}", file=sys.stderr)
        return []

    synced_files = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.endswith((".bin", ".done", ".json")):
            synced_files.append(line)
            print(f"  ✓ Synced: {line}")

    return synced_files


def watch_and_sync(
    host: str = "95.253.220.115",
    port: int = 62855,
    key_path: str = "~/.ssh/id_ed25519",
    interval: int = 60,
    remote_dir: str = "/workspace/truco-bot/models/",
    local_dir: str = "models/",
) -> None:
    """Continuously poll and sync new model checkpoints."""
    print(f"Starting continuous checkpoint watcher (polling every {interval}s)...")
    print(f"Target local directory: {Path(local_dir).resolve()}")
    try:
        while True:
            synced = sync_models(
                host=host,
                port=port,
                key_path=key_path,
                remote_dir=remote_dir,
                local_dir=local_dir,
            )
            if synced:
                print(f"  New artifacts transferred: {len(synced)}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync model checkpoints from remote Vast instance")
    parser.add_argument("--host", default=os.getenv("VAST_HOST", "95.253.220.115"), help="Remote host IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("VAST_PORT", "62855")), help="Remote SSH port")
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH identity key path")
    parser.add_argument("--remote-dir", default="/workspace/truco-bot/models/", help="Remote models directory")
    parser.add_argument("--local-dir", default="models/", help="Local models directory")
    parser.add_argument("--watch", action="store_true", help="Run in continuous watch mode")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    args = parser.parse_args()

    if args.watch:
        watch_and_sync(
            host=args.host,
            port=args.port,
            key_path=args.key,
            interval=args.interval,
            remote_dir=args.remote_dir,
            local_dir=args.local_dir,
        )
    else:
        synced = sync_models(
            host=args.host,
            port=args.port,
            key_path=args.key,
            remote_dir=args.remote_dir,
            local_dir=args.local_dir,
        )
        print(f"Sync complete. {len(synced)} file(s) updated.")


if __name__ == "__main__":
    main()
