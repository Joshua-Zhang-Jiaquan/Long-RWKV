"""Prove a local checkpoint's exact correspondence to a pinned BF16 Hub export.

Fetch only the safetensors header, then reconstruct the file digest by streaming
local tensor bytes in the remote offset order. Never download tensor payloads or
write an exported model. Default fetches/audits the header; --verify also hashes
all local tensors. Network access occurs only when --header is absent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

REVISION = '9d2b9042d7ab32e5d706479373a5437b1a29fb4b'
URL = f'https://huggingface.co/SII-Jiaquan/LACES-BiRWKV-DLM-2.9B-F2/resolve/{REVISION}/model.safetensors'
EXPECTED_SHA256 = '0e98d75dab14b16feffea4a6b692550cdd558b676d5058bcde0495b5b838d231'
EXPECTED_SIZE = 8183369960
EXPECTED_KEYS = 1952
MAX_HEADER_BYTES = 16 * 1024 * 1024
DEFAULT_CHECKPOINT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000/model.pt')
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8,
               'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}


def read_exact(stream, n):
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError('truncated safetensors header')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def fetch_header(url=URL, opener=urllib.request.urlopen):
    # A server may ignore Range. Bounded reads plus context-manager close still
    # prevent consuming its tensor payload. Keep original bytes including padding.
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{MAX_HEADER_BYTES + 7}',
                                                   'Accept-Encoding': 'identity'})
    with opener(request, timeout=60) as response:
        prefix = read_exact(response, 8)
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f'unsafe safetensors header length: {length}')
        header = prefix + read_exact(response, length)
        info = {'requested_url': url, 'resolved_url': response.geturl(),
                'http_status': response.status, 'bytes_read': len(header),
                'content_range': response.headers.get('Content-Range')}
    parse_header(header)
    return header, info


def parse_header(raw):
    if len(raw) < 8:
        raise ValueError('missing safetensors length prefix')
    length = struct.unpack('<Q', raw[:8])[0]
    if not 2 <= length <= MAX_HEADER_BYTES or len(raw) != length + 8:
        raise ValueError('header byte length mismatch')
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f'duplicate header key: {key}')
            out[key] = value
        return out
    header = json.loads(raw[8:], object_pairs_hook=unique_pairs)
    if not isinstance(header, dict):
        raise ValueError('safetensors header must be an object')
    entries = []
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(spec, dict) or spec.get('dtype') not in DTYPE_BYTES:
            raise ValueError(f'unsupported tensor dtype: {name}')
        shape, offsets = spec.get('shape'), spec.get('data_offsets')
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'invalid shape: {name}')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise ValueError(f'invalid offsets: {name}')
        elements = 1
        for n in shape:
            elements *= n
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[spec['dtype']]:
            raise ValueError(f'tensor byte count mismatch: {name}')
        entries.append((name, spec))
    entries.sort(key=lambda x: (x[1]['data_offsets'][0], x[1]['data_offsets'][1]))
    end = 0
    for name, spec in entries:
        if spec['data_offsets'][0] != end:
            raise ValueError(f'noncontiguous or overlapping offsets: {name}')
        end = spec['data_offsets'][1]
    return entries, len(raw) + end


def reconstruct_digest(raw, state, *, chunk_bytes=8 * 1024 * 1024):
    """Hash original header and converted local tensors without an export file."""
    import sys
    import torch
    if sys.byteorder != 'little':
        raise ValueError('safetensors reconstruction requires a little-endian host')
    if chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be positive')
    entries, total_size = parse_header(raw)
    if set(state) != {name for name, _ in entries}:
        missing = sorted({name for name, _ in entries} - set(state))
        extra = sorted(set(state) - {name for name, _ in entries})
        raise ValueError(f'checkpoint keys mismatch: missing={missing[:8]} extra={extra[:8]}')
    dtypes = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
              'F64': torch.float64, 'I64': torch.int64, 'I32': torch.int32,
              'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
    # Check all shapes before doing a potentially expensive conversion/hash pass.
    for name, spec in entries:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != spec['shape']:
            raise ValueError(f'checkpoint shape mismatch: {name}')
        if tensor.device.type != 'cpu' or tensor.layout != torch.strided:
            raise ValueError(f'expected dense CPU tensor: {name}')
    h = hashlib.sha256(raw)
    bytes_hashed = len(raw)
    for name, spec in entries:
        # Largest allocation is one converted tensor, not the entire BF16 model.
        tensor = state[name].detach().to(dtype=dtypes[spec['dtype']]).contiguous()
        byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
        buffer = memoryview(byte_view)
        for start in range(0, len(buffer), chunk_bytes):
            block = buffer[start:start + chunk_bytes]
            h.update(block)
            bytes_hashed += len(block)
        del buffer, byte_view, tensor
    if bytes_hashed != total_size:
        raise ValueError('reconstructed file size differs from safetensors offsets')
    return {'sha256': h.hexdigest(), 'bytes': bytes_hashed, 'tensor_keys': len(entries)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--header', type=Path, help='existing original 8-byte prefix + JSON header; no network')
    ap.add_argument('--out', type=Path, default=Path('results/f2_export_audit'))
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--verify', action='store_true', help='hash full export from mmap local checkpoint')
    args = ap.parse_args(argv)
    if args.header:
        raw = args.header.read_bytes()
        acquisition = {'source': str(args.header.resolve()), 'network': False}
    else:
        raw, acquisition = fetch_header()
    entries, size = parse_header(raw)
    if len(entries) != EXPECTED_KEYS or size != EXPECTED_SIZE:
        raise ValueError(f'pinned export structure mismatch: keys={len(entries)} bytes={size}')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'header.bin').write_bytes(raw)
    audit = {'revision': REVISION, 'url': URL, 'expected_sha256': EXPECTED_SHA256,
             'expected_size': EXPECTED_SIZE, 'header_sha256': hashlib.sha256(raw).hexdigest(),
             'header_bytes': len(raw), 'tensor_keys': len(entries), 'acquisition': acquisition,
             'verified': False, 'meaning': 'header structure only; tensor correspondence not checked'}
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if args.verify:
        import torch
        torch.set_num_threads(1)
        state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
        result = reconstruct_digest(raw, state)
        audit.update({'checkpoint': str(args.checkpoint.resolve()), 'reconstructed': result,
                      'verified': result['sha256'] == EXPECTED_SHA256,
                      'meaning': 'SHA256 equality proves exact bytes of pinned remote export after declared per-tensor dtype conversion'})
        (args.out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        if not audit['verified']:
            raise SystemExit('export digest mismatch: local tensors do not reconstruct the pinned artifact')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
