import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np

from usv_vla.inspect_weights import abstract_model_reference, compare_model, inspect_weights, read_npy_header


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value)
    return stream.getvalue()


class WeightInspectionTests(unittest.TestCase):
    def test_tiny_npz_headers_hash_crc_without_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            np.savez(path, **{"params/llm/embedder/input_embedding": np.zeros((2, 3), np.float32),
                             "params/img/embedding/kernel": np.ones((1, 2), np.float32)})
            with mock.patch("numpy.load", side_effect=AssertionError("Array materialization forbidden")):
                result = inspect_weights(path, crc=True)
            self.assertEqual(result["status"], "ok", result["errors"])
            self.assertEqual(result["total_array_bytes"], 32)
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["verification"]["zip_crc"], "ok")
            self.assertEqual(result["arrays"]["params/img/embedding/kernel"]["shape"], [1, 2])

    def test_header_read_does_not_consume_payload(self):
        encoded = _npy(np.ones((100, 50), np.float32))
        stream = io.BytesIO(encoded)
        result = read_npy_header(stream)
        self.assertEqual(stream.tell(), result["header_bytes"])
        self.assertLess(stream.tell(), 1024)
        self.assertEqual(result["array_bytes"], 20_000)

    def test_expected_digest_and_byte_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            np.savez(path, **{"params/llm/x": np.zeros(1), "params/img/y": np.zeros(1)})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = inspect_weights(path, expected_sha256=digest, expected_bytes=path.stat().st_size)
            self.assertEqual(result["status"], "ok", result["errors"])
            self.assertEqual(result["verification"]["sha256"], "matched_expected_digest")
            result = inspect_weights(path, expected_sha256="0" * 64, expected_bytes=1)
            self.assertEqual(result["status"], "error")
            self.assertTrue(any("SHA-256 mismatch" in error for error in result["errors"]))
            self.assertTrue(any("File size mismatch" in error for error in result["errors"]))
            self.assertEqual(inspect_weights(path, expected_sha256="not-a-digest")["status"], "error")
            self.assertEqual(inspect_weights(path, sha256=False, expected_sha256=digest)["status"], "error")

    def test_all_supported_header_versions(self):
        for version in ((1, 0), (2, 0), (3, 0)):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.zeros((2,), np.float32), version=version)
            stream.seek(0)
            self.assertEqual(read_npy_header(stream)["npy_version"], list(version))

    def test_duplicate_object_missing_branch_and_wrong_payload_rejected(self):
        cases = {
            "object": [("params/llm/object.npy", _npy(np.array([object()], dtype=object)))],
            "duplicate": [("params/llm/x.npy", _npy(np.zeros(1))), ("params/llm/x.npy", _npy(np.zeros(1)))],
            "missing": [("params/llm/x.npy", _npy(np.zeros(1)))],
            "truncated": [("params/llm/x.npy", _npy(np.zeros(1))[:-1]), ("params/img/y.npy", _npy(np.zeros(1)))],
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, entries in cases.items():
                path = Path(directory) / (name + ".npz")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(path, "w") as archive:
                        for key, data in entries:
                            archive.writestr(key, data)
                result = inspect_weights(path, sha256=False)
                self.assertEqual(result["status"], "error", name)
                self.assertTrue(result["errors"], name)
            self.assertIn("Object arrays", inspect_weights(Path(directory) / "object.npz", sha256=False)["errors"][0])
            self.assertTrue(any("Duplicate" in error for error in inspect_weights(Path(directory) / "duplicate.npz", sha256=False)["errors"]))

    def test_model_reuse_and_expected_new_parameters(self):
        source = {"params/llm/x": {"shape": [2, 3], "dtype": "float16", "array_bytes": 12},
                  "params/img/y": {"shape": [4], "dtype": "float32", "array_bytes": 16}}
        reference = {"PaliGemma/llm/x": {"shape": [2, 3], "dtype": "float32"},
                     "PaliGemma/img/y": {"shape": [4], "dtype": "float32"},
                     "PaliGemma/llm/layers/mlp_1/linear": {"shape": [8], "dtype": "float32"},
                     "action_in_proj/kernel": {"shape": [2, 1024], "dtype": "float32"}}
        result = compare_model(source, reference)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reused_parameter_count"], 2)
        self.assertEqual(result["expected_new_parameter_count"], 2)
        self.assertEqual(len(result["dtype_casts_performed_by_loader"]), 1)
        reference["PaliGemma/llm/x"]["shape"] = [3, 2]
        self.assertEqual(compare_model(source, reference)["status"], "error")
        reference["PaliGemma/llm/required"] = {"shape": [1], "dtype": "float32"}
        result = compare_model(source, reference)
        self.assertIn("PaliGemma/llm/required", result["missing_backbone_parameter_keys"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(compare_model({}, reference)["reused_parameter_count"], 0)
        self.assertEqual(compare_model({}, reference)["status"], "error")

    def test_corrupt_archive_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            path.write_text("not an npz")
            self.assertEqual(inspect_weights(path, sha256=False)["status"], "error")

    def test_real_model_reference_is_abstract(self):
        try:
            import flax.nnx  # noqa: F401
            import jax  # noqa: F401
        except ImportError:
            self.skipTest("Run with the model Python for the full abstract model check")
        reference = abstract_model_reference()
        self.assertEqual(reference["action_in_proj/kernel"]["shape"], [2, 1024])
        self.assertEqual(reference["action_out_proj/kernel"]["shape"], [1024, 2])
        self.assertEqual(reference["PaliGemma/llm/embedder/input_embedding"]["shape"], [257152, 2048])
        self.assertGreater(len(reference), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
