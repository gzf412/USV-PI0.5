"""Inspect a PaliGemma JAX NPZ without materializing checkpoint arrays.

NPY headers establish names, shapes and storage sizes. Optional SHA-256 and ZIP
CRC checks stream the file; neither loads the checkpoint into model memory.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any
import zipfile

import numpy as np


_MAX_HEADER_BYTES = 1024 * 1024
_CHUNK_BYTES = 8 * 1024 * 1024
_CORE_BRANCHES = ("params/llm/", "params/img/")


def _read_exact(stream: Any, size: int) -> bytes:
    result = stream.read(size)
    if len(result) != size:
        raise ValueError("Truncated NPY header")
    return result


def read_npy_header(stream: Any) -> dict[str, Any]:
    """Read only a bounded NPY header (versions 1, 2 and 3)."""
    if _read_exact(stream, 6) != b"\x93NUMPY":
        raise ValueError("ZIP member is not an NPY array")
    version = tuple(_read_exact(stream, 2))
    if version == (1, 0):
        length_bytes, length_format, encoding = 2, "<H", "latin1"
    elif version in ((2, 0), (3, 0)):
        length_bytes, length_format = 4, "<I"
        encoding = "utf8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    header_length = struct.unpack(length_format, _read_exact(stream, length_bytes))[0]
    if header_length > _MAX_HEADER_BYTES:
        raise ValueError(f"NPY header exceeds {_MAX_HEADER_BYTES} byte limit")
    header = ast.literal_eval(_read_exact(stream, header_length).decode(encoding))
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("Invalid NPY header fields")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("Invalid NPY shape")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("Invalid NPY fortran_order")
    dtype = np.dtype(header["descr"])
    if dtype.hasobject:
        raise ValueError("Object arrays are forbidden in model checkpoints")
    return {
        "npy_version": list(version),
        "shape": list(shape),
        "dtype": str(dtype),
        "dtype_descr": header["descr"],
        "fortran_order": header["fortran_order"],
        "array_bytes": math.prod(shape) * dtype.itemsize,
        "header_bytes": 8 + length_bytes + header_length,
    }


def _is_action_expert(key: str) -> bool:
    # Gemma's second module is the new action expert. This is the same _1
    # naming convention used by Pi0Config.get_freeze_filter().
    return key.startswith("PaliGemma/llm/") and any(part.endswith("_1") for part in key.split("/"))


def abstract_model_reference() -> dict[str, dict[str, Any]]:
    """Trace the real configuration on CPU and return shape-only parameters."""
    # Set this before JAX's first import; normal CLI use cannot allocate a GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import flax.nnx as nnx
    import flax.traverse_util
    import jax
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, state_dim=33, action_dim=2, action_horizon=10, max_token_len=384)
    with jax.default_device(jax.devices("cpu")[0]):
        model = nnx.eval_shape(config.create, jax.random.key(0))
        flat = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    # Fail loudly if future model changes accidentally turn abstract tracing
    # into materialization. ShapeDtypeStruct leaves contain no parameter data.
    if not flat or any(not isinstance(value, jax.ShapeDtypeStruct) for value in flat.values()):
        raise RuntimeError("Model reference must consist only of abstract ShapeDtypeStruct leaves")
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in flat.items()}


def compare_model(arrays: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mirror PaliGemmaWeightLoader's key selection, without reading values."""
    loaded = {"PaliGemma/" + key[len("params/"):]: value for key, value in arrays.items() if key.startswith("params/")}
    reused = sorted(set(loaded) & set(reference))
    missing = sorted(set(reference) - set(loaded))
    expected_new = [key for key in missing if not key.startswith("PaliGemma/") or _is_action_expert(key)]
    missing_backbone = [key for key in missing if key not in expected_new]
    mismatches = []
    casts = []
    for key in reused:
        source, target = loaded[key], reference[key]
        if list(source["shape"]) != list(target["shape"]):
            mismatches.append({"key": key, "source_shape": source["shape"], "model_shape": target["shape"]})
        if source["dtype"] != target["dtype"]:
            casts.append({"key": key, "source_dtype": source["dtype"], "model_dtype": target["dtype"]})
    branches = {branch: sum(key.startswith("PaliGemma/" + branch + "/") for key in reused) for branch in ("llm", "img")}
    errors = []
    if not reused:
        errors.append("No checkpoint parameters match the model; initialization would silently keep random parameters")
    for branch, count in branches.items():
        if count == 0:
            errors.append(f"No reusable PaliGemma {branch} parameters")
    if missing_backbone:
        errors.append(f"Missing {len(missing_backbone)} PaliGemma backbone parameters")
    if mismatches:
        errors.append(f"Shape mismatch in {len(mismatches)} reusable parameters")
    return {
        "status": "error" if errors else "ok",
        "method": "CPU nnx.eval_shape; headers only; same params/ to PaliGemma/ key mapping as PaliGemmaWeightLoader",
        "model_config": {"pi05": True, "state_dim": 33, "action_dim": 2, "action_horizon": 10, "max_token_len": 384},
        "reference_parameter_count": len(reference),
        "reused_parameter_count": len(reused),
        "reused_parameter_keys": reused,
        "reused_branches": branches,
        "reused_array_bytes": sum(loaded[key]["array_bytes"] for key in reused),
        "expected_new_parameter_count": len(expected_new),
        "expected_new_parameter_keys": expected_new,
        "missing_backbone_parameter_keys": missing_backbone,
        "shape_mismatches": mismatches,
        "dtype_casts_performed_by_loader": casts,
        "ignored_source_parameter_keys": sorted(set(loaded) - set(reference)),
        "errors": errors,
        "limitations": [
            "Does not allocate, load, or run the numeric model and does not test optimizer memory or GPU capacity.",
            "Headers do not establish finite values, correct semantics, or trained performance.",
            "New action expert and action/time projection parameters remain randomly initialized by the model.",
        ],
    }


def inspect_weights(path: Path, *, sha256: bool = True, crc: bool = False, check_model: bool = False,
                    expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1, "path": str(path), "status": "error", "errors": [], "arrays": {},
        "verification": {"header_check": "not_run", "sha256": "not_run", "zip_crc": "not_run"},
    }
    errors = report["errors"]
    try:
        source_stat = path.stat()
        report["file_bytes"] = source_stat.st_size
        if expected_bytes is not None:
            report["expected_bytes"] = expected_bytes
            report["verification"]["expected_bytes"] = "ok" if source_stat.st_size == expected_bytes else "error"
            if source_stat.st_size != expected_bytes:
                errors.append(f"File size mismatch: got {source_stat.st_size}, expected {expected_bytes}")
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
                raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")
            report["expected_sha256"] = expected_sha256
            if not sha256:
                raise ValueError("Cannot compare an expected SHA-256 when --no-sha256 is selected")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = Counter(member.filename for member in members)
            duplicate_names = sorted(name for name, count in names.items() if count > 1)
            if duplicate_names:
                errors.append(f"Duplicate ZIP members: {duplicate_names}")
            for member in members:
                if member.is_dir():
                    continue
                name = member.filename
                if not name.endswith(".npy"):
                    errors.append(f"Unexpected non-NPY member: {name}")
                    continue
                key = name[:-4]
                if any(part in ("", ".", "..") for part in key.split("/")):
                    errors.append(f"Invalid parameter key: {key}")
                    continue
                if key in report["arrays"]:
                    errors.append(f"Duplicate parameter key: {key}")
                    continue
                try:
                    with archive.open(member) as stream:
                        array = read_npy_header(stream)
                    array.update({"zip_member": name, "zip_uncompressed_bytes": member.file_size,
                                  "zip_compressed_bytes": member.compress_size, "zip_crc32": f"{member.CRC:08x}"})
                    report["arrays"][key] = array
                    if array["header_bytes"] + array["array_bytes"] != member.file_size:
                        errors.append(f"NPY payload size mismatch: {key}")
                except (ValueError, SyntaxError, UnicodeError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
                    errors.append(f"Invalid NPY {key}: {exc}")
            for branch in _CORE_BRANCHES:
                if not any(key.startswith(branch) for key in report["arrays"]):
                    errors.append(f"Missing core source branch: {branch}")
            report["verification"]["header_check"] = "error" if errors else "ok"
            if crc:
                bad_member = archive.testzip()
                report["verification"]["zip_crc"] = "error" if bad_member else "ok"
                if bad_member:
                    errors.append(f"ZIP CRC check failed: {bad_member}")
        if sha256:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(_CHUNK_BYTES):
                    digest.update(chunk)
            report["sha256"] = digest.hexdigest()
            if expected_sha256 is None:
                report["verification"]["sha256"] = "computed_not_compared_to_upstream"
            else:
                matched = report["sha256"] == expected_sha256
                report["verification"]["sha256"] = "matched_expected_digest" if matched else "error"
                if not matched:
                    errors.append(f"SHA-256 mismatch: got {report['sha256']}, expected {expected_sha256}")
        if check_model:
            compatibility = compare_model(report["arrays"], abstract_model_reference())
            report["model_compatibility"] = compatibility
            errors.extend(compatibility["errors"])
        after_stat = path.stat()
        if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) != (
            after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns
        ):
            errors.append("Checkpoint changed during inspection; do not inspect an in-progress download")
    except (OSError, ValueError, RuntimeError, ImportError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["parameter_count"] = len(report["arrays"])
    report["total_array_bytes"] = sum(array["array_bytes"] for array in report["arrays"].values())
    report["status"] = "error" if errors else "ok"
    report["limitations"] = [
        "SHA-256 is a local fingerprint until compared with a trusted upstream digest.",
        "A matched trusted SHA-256 verifies all archive bytes; --crc separately checks ZIP member CRCs. Header checks alone do not verify payload integrity.",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Downloaded PaliGemma JAX .npz")
    parser.add_argument("--report", type=Path, help="Full JSON report (default: PATH.inspection.json)")
    parser.add_argument("--sha256", action=argparse.BooleanOptionalAction, default=True, help="Stream a local SHA-256 fingerprint (default: on)")
    parser.add_argument("--crc", action="store_true", help="Additionally stream/decompress every ZIP member and check its CRC")
    parser.add_argument("--check-model", action="store_true", help="Trace the actual USV pi05 model on CPU and compare reused parameter shapes")
    parser.add_argument("--expected-sha256", help="Fail unless the streamed digest matches this trusted SHA-256")
    parser.add_argument("--expected-bytes", type=int, help="Fail unless the file size matches this trusted byte count")
    args = parser.parse_args()
    report = inspect_weights(args.path, sha256=args.sha256, crc=args.crc, check_model=args.check_model,
                             expected_sha256=args.expected_sha256, expected_bytes=args.expected_bytes)
    output = args.report or Path(str(args.path) + ".inspection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    summary = {key: report[key] for key in ("status", "parameter_count", "total_array_bytes", "verification")}
    if "model_compatibility" in report:
        model = report["model_compatibility"]
        summary["model_compatibility"] = {key: model[key] for key in ("status", "reused_parameter_count", "expected_new_parameter_count")}
    summary["errors"] = report["errors"][:10]
    summary["error_count"] = len(report["errors"])
    summary["report"] = str(output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
