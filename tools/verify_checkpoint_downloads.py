"""Verify anonymous release metadata and small model download ranges."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def get(url, limit=None):
    headers = {'User-Agent': 'Long-RWKV-release-verifier'}
    if limit:
        headers['Range'] = f'bytes=0-{limit-1}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as source:
        return source.status, source.read() if limit is None else source.read(limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-exports', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((ROOT / 'models_release/EXPORTS.json').read_text())
    api = f"https://api.github.com/repos/{registry['repository']}/releases/{registry['release_id']}"
    _, raw = get(api)
    release = json.loads(raw)
    assert not release['draft'] and release['tag_name'] == registry['release_tag']
    stamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
    _, asset_raw = get(api + f'/assets?per_page=100&audit={stamp}')
    assets = {a['name']: a for a in json.loads(asset_raw)}

    def check(item):
        role, record = item
        asset = assets[record['asset_name']]
        assert asset['state'] == 'uploaded' and asset['size'] == record['bytes']
        assert asset['digest'] == 'sha256:' + record['export_sha256']
        status, prefix = get(record['download_url'], 64)
        with (args.local_exports / record['asset_name']).open('rb') as stream:
            assert prefix == stream.read(64) and len(prefix) == 64
        return dict(role=role, asset_id=asset['id'], bytes=asset['size'],
                    server_digest=asset['digest'], anonymous_http_status=status,
                    anonymous_first64_matches=True, download_url=record['download_url'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        models = list(pool.map(check, registry['exports'].items()))
    metadata = []
    for name in ('EXPORTS.json', 'README.md', 'APACHE-2.0.txt'):
        asset = assets[name]
        download_url = asset['browser_download_url'] + '?sha256=' + asset['digest'].removeprefix('sha256:')
        _, data = get(download_url)
        digest = hashlib.sha256(data).hexdigest()
        assert data == (ROOT / 'models_release' / name).read_bytes(), name
        assert asset['digest'] == 'sha256:' + digest, name
        metadata.append(dict(name=name, sha256=digest, anonymous_download_verified=True,
                             download_url=download_url))
    proof = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 release_id=release['id'], release_url=release['html_url'],
                 release_tag=release['tag_name'], published_at=release['published_at'],
                 draft=False, target_commitish=release['target_commitish'], models=models,
                 metadata=metadata, checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Anonymous API sizes/digests and 64-byte model ranges verified; metadata downloaded in full. Full model identity uses the previously verified local exports and server SHA256 values. No new neural inference.',
                 listing_note='Uses release-by-ID and its assets endpoint; the tag endpoint returned an incomplete asset listing. Metadata probes include digest query strings to avoid a stale replaced README cache.')
    args.out.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(status='passed', models=len(models), metadata=len(metadata), release=release['html_url'])))


if __name__ == '__main__':
    main()
