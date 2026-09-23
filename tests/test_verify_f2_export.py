"""Tiny synthetic serialization tests; no model or network access."""
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence import verify_f2_export as V


def fixture():
    state = {'z': torch.tensor([1.5, -2.25], dtype=torch.float32),
             'a': torch.tensor([[3]], dtype=torch.int64)}
    h = {'__metadata__': {'format': 'pt'},
         'a': {'dtype': 'I64', 'shape': [1, 1], 'data_offsets': [4, 12]},
         'z': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}
    body = json.dumps(h).encode()
    body += b' ' * (-len(body) % 8)
    raw = struct.pack('<Q', len(body)) + body
    payload = state['z'].to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    payload += state['a'].view(torch.uint8).numpy().tobytes()
    return state, raw, payload


class ExportTests(unittest.TestCase):
    def test_exact_hash_offset_order_and_dtype(self):
        state, raw, payload = fixture()
        result = V.reconstruct_digest(raw, state, chunk_bytes=3)
        self.assertEqual(result['sha256'], hashlib.sha256(raw + payload).hexdigest())
        self.assertEqual(result['bytes'], len(raw) + len(payload))
        self.assertEqual(result['tensor_keys'], 2)

    def test_shape_and_keys_rejected(self):
        state, raw, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'keys mismatch'):
            V.reconstruct_digest(raw, {'z': state['z']})
        state['a'] = state['a'].reshape(1)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            V.reconstruct_digest(raw, state)

    def test_noncontiguous_offsets_rejected(self):
        _, raw, _ = fixture()
        header = json.loads(raw[8:])
        header['a']['data_offsets'] = [6, 14]
        data = json.dumps(header).encode()
        with self.assertRaisesRegex(ValueError, 'noncontiguous'):
            V.parse_header(struct.pack('<Q', len(data)) + data)

    def test_ignores_range_server_closed_before_payload(self):
        _, raw, payload = fixture()
        class Response(io.BytesIO):
            status = 200
            headers = {}
            read_bytes = 0
            def geturl(self):
                return 'https://example.test/model'
            def read(self, n=-1):
                out = super().read(n)
                self.read_bytes += len(out)
                return out
        response = Response(raw + payload * 100)
        got, info = V.fetch_header(opener=lambda *a, **kw: response)
        self.assertEqual(got, raw)
        self.assertEqual(response.read_bytes, len(raw))
        self.assertTrue(response.closed)
        self.assertEqual(info['http_status'], 200)

    def test_oversized_header_refused_and_closed(self):
        class Response(io.BytesIO):
            pass
        response = Response(struct.pack('<Q', V.MAX_HEADER_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            V.fetch_header(opener=lambda *a, **kw: response)
        self.assertTrue(response.closed)


if __name__ == '__main__':
    unittest.main()
