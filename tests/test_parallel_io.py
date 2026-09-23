import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from lrwkv_evidence import parallel_io as P


class ParallelIOTests(unittest.TestCase):
    def test_order_exact_bytes_and_bounded_input(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / f'{i}.json' for i in range(10)]
            for i, path in enumerate(paths):
                path.write_bytes(f'{{ "i": {i} }}\n'.encode())
            consumed = []
            def source():
                for path in paths:
                    consumed.append(path)
                    yield path
            fourth_started = threading.Event()
            original = P._read_json
            def delayed(path):
                if path == paths[3]:
                    fourth_started.set()
                if path == paths[0]:
                    if not fourth_started.wait(5):
                        raise AssertionError('independent reads did not run concurrently')
                return original(path)
            with patch.object(P, '_read_json', side_effect=delayed):
                reader = P.read_json_parallel(source(), max_workers=4, max_inflight=4)
                first = next(reader)
                self.assertEqual(len(consumed), 4)
                results = [first, *reader]
            self.assertEqual([r[0] for r in results], paths)
            for i, (path, raw, value) in enumerate(results):
                self.assertEqual(raw, path.read_bytes())
                self.assertEqual(value, {'i': i})

    def test_invalid_json_propagates_without_skipping(self):
        with tempfile.TemporaryDirectory() as root:
            good, bad = Path(root) / 'good.json', Path(root) / 'bad.json'
            good.write_text('{}')
            bad.write_text('{broken')
            reader = P.read_json_parallel([good, bad, good])
            self.assertEqual(next(reader)[2], {})
            with self.assertRaises(json.JSONDecodeError):
                next(reader)
