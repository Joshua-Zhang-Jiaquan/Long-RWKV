"""Download a released inference export and verify its exact container digest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path, record):
    if path.stat().st_size != record['bytes']:
        raise ValueError(f'Checkpoint size mismatch: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != record['export_sha256']:
        raise ValueError(f'Checkpoint SHA256 mismatch: {path}')
    return digest


def main():
    registry = json.loads((ROOT / 'models_release/EXPORTS.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=sorted(registry['exports']), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    record = registry['exports'][args.role]
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / record['asset_name']
    if not target.exists():
        if args.verify_only:
            raise FileNotFoundError(target)
        temporary = target.with_suffix(target.suffix + '.download')
        # Exclusive creation preserves incomplete downloads for inspection.
        with temporary.open('xb') as dest:
            with urllib.request.urlopen(record['download_url'], timeout=60) as source:
                while block := source.read(8 * 1024 * 1024):
                    dest.write(block)
        verify(temporary, record)
        os.replace(temporary, target)
    digest = verify(target, record)
    print(json.dumps(dict(role=args.role, path=str(target), sha256=digest,
                         bytes=record['bytes'], status='verified')))


if __name__ == '__main__':
    main()
