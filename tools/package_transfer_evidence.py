"""Package complete frozen raw panels without changing their original identities."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'results/predictive_transfer')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/predictive_transfer/raw')
    args = parser.parse_args()
    folder, out = args.folder.resolve(), args.out.resolve()
    out.relative_to(ROOT)  # Portable index paths are repository-relative.
    if out.exists():
        raise FileExistsError(out)
    seal_path = folder / 'PREDICTION_SEAL.json'
    seal = json.loads(seal_path.read_text())
    report = json.loads((folder / 'report.json').read_text())
    if report['prediction_seal_sha256'] != sha(seal_path):
        raise ValueError('Report/seal mismatch')
    if len(seal['predictions']) != 6 or not report['gates']['all_six_complete']:
        raise ValueError('Incomplete experiment')
    expected = dict(report['raw_sha256'])
    for artifact in seal['predictions'].values():
        path = ROOT / artifact['path']
        if sha(path) != artifact['sha256']:
            raise ValueError('Prediction seal broken')
        for original, digest in json.loads(path.read_text())['calibration_raw_sha256'].items():
            if original in expected:
                raise ValueError('Repeated raw path across source/target panels')
            expected[original] = digest
    if len(expected) != 96:
        raise ValueError('Expected 6 lineages x 2 panels x 8 ranks')
    pending = out.with_name(out.name + '.preparing')
    pending.mkdir(parents=True, exist_ok=False)
    records = []
    for i, (original, digest) in enumerate(sorted(expected.items())):
        source = Path(original)
        if sha(source) != digest:
            raise ValueError('Raw hash changed: ' + original)
        name = f'rank_file_{i:03d}.jsonl.gz'
        dest = pending / name
        with source.open('rb') as stream, dest.open('xb') as encoded:
            with gzip.GzipFile(filename='', mode='wb', fileobj=encoded, mtime=0) as zipped:
                shutil.copyfileobj(stream, zipped)
        with gzip.open(dest, 'rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError('Compression roundtrip mismatch')
        records.append(dict(original_path=original, path=str((out / name).relative_to(ROOT)),
                            sha256=digest, gzip_sha256=sha(dest),
                            bytes=source.stat().st_size, gzip_bytes=dest.stat().st_size))
    index = dict(schema_version=1, prediction_seal_sha256=sha(seal_path),
                 report_sha256=sha(folder / 'report.json'), files=records,
                 scope='Lossless raw-panel packaging; independent scientific audit is separate.')
    (pending / 'INDEX.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    os.rename(pending, out)
    print(json.dumps(dict(status='packaged', rank_files=len(records), index=str(out / 'INDEX.json'))))


if __name__ == '__main__':
    main()
