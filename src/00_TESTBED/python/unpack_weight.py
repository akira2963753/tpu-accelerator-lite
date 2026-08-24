#!/usr/bin/env python3
"""Restore a hexadecimal weight file from gzip-compressed Base85 parts."""

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


FORMAT_VERSION = "tpu-dat-b85-v1"
NEWLINES = {
    "crlf": b"\r\n",
    "lf": b"\n",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore a hexadecimal .dat file from Base85 transfer parts."
    )
    parser.add_argument("transfer_dir", type=Path, help="directory containing manifest.json")
    parser.add_argument("-o", "--output", type=Path, help="restored output path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    return parser.parse_args()


def require_integer(manifest, key, minimum=0):
    value = manifest.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"manifest field '{key}' must be an integer >= {minimum}")
    return value


def load_manifest(path):
    try:
        manifest = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest: {error}") from error

    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported transfer format: {manifest.get('format')!r}")
    if manifest.get("compression") != "gzip":
        raise ValueError("manifest compression must be 'gzip'")
    if manifest.get("encoding") != "base85":
        raise ValueError("manifest encoding must be 'base85'")
    if manifest.get("newline") not in NEWLINES:
        raise ValueError("manifest newline must be 'crlf' or 'lf'")
    if manifest.get("hex_case") not in ("lower", "upper"):
        raise ValueError("manifest hex_case must be 'lower' or 'upper'")
    if not isinstance(manifest.get("final_newline"), bool):
        raise ValueError("manifest final_newline must be a boolean")

    source_name = manifest.get("source_name")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise ValueError("manifest source_name must be a plain file name")

    require_integer(manifest, "source_size", 1)
    require_integer(manifest, "raw_size", 1)
    require_integer(manifest, "compressed_size", 1)
    require_integer(manifest, "payload_chars", 1)
    require_integer(manifest, "part_count", 1)
    require_integer(manifest, "line_count", 1)
    line_hex_chars = require_integer(manifest, "line_hex_chars", 2)
    if line_hex_chars % 2:
        raise ValueError("manifest line_hex_chars must be even")

    for key in ("source_sha256", "compressed_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"manifest field '{key}' must be a SHA-256 hex digest")

    return manifest


def read_payload(transfer_dir, part_count):
    parts = []
    for index in range(part_count):
        path = transfer_dir / f"part{index:03d}.b85"
        if not path.is_file():
            raise ValueError(f"missing transfer part: {path.name}")
        try:
            text = path.read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read {path.name}: {error}") from error
        parts.append("".join(text.split()))
    return "".join(parts)


def restore_source(raw_data, manifest):
    line_bytes = manifest["line_hex_chars"] // 2
    expected_raw_size = manifest["line_count"] * line_bytes
    if len(raw_data) != expected_raw_size:
        raise ValueError(
            f"raw layout mismatch: expected {expected_raw_size:,} bytes, "
            f"got {len(raw_data):,}"
        )

    rows = []
    for offset in range(0, len(raw_data), line_bytes):
        row = raw_data[offset:offset + line_bytes].hex()
        rows.append(row.upper() if manifest["hex_case"] == "upper" else row)

    newline = NEWLINES[manifest["newline"]]
    source_data = newline.join(row.encode("ascii") for row in rows)
    if manifest["final_newline"]:
        source_data += newline
    return source_data


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    transfer_dir = args.transfer_dir.resolve()
    manifest_path = transfer_dir / "manifest.json"

    try:
        manifest = load_manifest(manifest_path)
        payload = read_payload(transfer_dir, manifest["part_count"])

        if len(payload) != manifest["payload_chars"]:
            raise ValueError(
                f"Base85 length mismatch: expected {manifest['payload_chars']:,}, "
                f"got {len(payload):,}"
            )

        try:
            compressed = base64.b85decode(payload.encode("ascii"))
        except (ValueError, binascii.Error) as error:
            raise ValueError(f"invalid Base85 payload: {error}") from error

        if len(compressed) != manifest["compressed_size"]:
            raise ValueError("compressed size does not match manifest")
        compressed_sha256 = hashlib.sha256(compressed).hexdigest()
        if compressed_sha256 != manifest["compressed_sha256"]:
            raise ValueError("compressed SHA-256 does not match manifest")

        try:
            raw_data = gzip.decompress(compressed)
        except (OSError, EOFError) as error:
            raise ValueError(f"gzip decompression failed: {error}") from error

        if len(raw_data) != manifest["raw_size"]:
            raise ValueError("raw size does not match manifest")

        source_data = restore_source(raw_data, manifest)
        if len(source_data) != manifest["source_size"]:
            raise ValueError("restored source size does not match manifest")

        source_sha256 = hashlib.sha256(source_data).hexdigest()
        if source_sha256 != manifest["source_sha256"]:
            raise ValueError("restored source SHA-256 does not match manifest")
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    output = args.output
    if output is None:
        output = transfer_dir.parent / manifest["source_name"]
    output = output.resolve()

    if output.exists() and not args.force:
        raise SystemExit(f"error: output already exists: {output} (use --force to overwrite)")

    write_atomic(output, source_data)

    print(f"restored : {output}")
    print(f"size     : {len(source_data):,} bytes")
    print(f"rows     : {manifest['line_count']:,}")
    print(f"SHA-256  : {source_sha256}")
    print("status   : verified")


if __name__ == "__main__":
    main()
