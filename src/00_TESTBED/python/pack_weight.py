#!/usr/bin/env python3
"""Pack a fixed-width hexadecimal weight file for text-only transfer."""

import argparse
import base64
import hashlib
import json
import string
import textwrap
import zlib
from pathlib import Path


FORMAT_VERSION = "tpu-dat-b85-v1"
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_WRAP_WIDTH = 100
GZIP_STRATEGIES = (
    ("default", zlib.Z_DEFAULT_STRATEGY),
    ("filtered", zlib.Z_FILTERED),
    ("rle", zlib.Z_RLE),
    ("huffman", zlib.Z_HUFFMAN_ONLY),
    ("fixed", zlib.Z_FIXED),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a hexadecimal .dat file into gzip-compressed Base85 parts."
    )
    parser.add_argument("input", type=Path, help="input hexadecimal .dat file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="output directory (default: <input_stem>_transfer)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Base85 characters per part (default: %(default)s)",
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=DEFAULT_WRAP_WIDTH,
        help="display columns inside each part (default: %(default)s)",
    )
    return parser.parse_args()


def split_rows(data):
    if b"\r\n" in data:
        newline = b"\r\n"
        newline_name = "crlf"
        residue = data.replace(newline, b"")
        if b"\r" in residue or b"\n" in residue:
            raise ValueError("mixed newline styles are not supported")
    elif b"\n" in data:
        newline = b"\n"
        newline_name = "lf"
        if b"\r" in data:
            raise ValueError("mixed newline styles are not supported")
    elif b"\r" in data:
        raise ValueError("CR-only newline style is not supported")
    else:
        newline = b"\n"
        newline_name = "lf"

    final_newline = data.endswith(newline)
    body = data[:-len(newline)] if final_newline else data
    rows = body.split(newline)

    if not rows or rows == [b""]:
        raise ValueError("input file is empty")
    if any(not row for row in rows):
        raise ValueError("blank rows are not supported")

    return rows, newline_name, final_newline


def decode_hex_rows(rows):
    line_hex_chars = len(rows[0])
    if line_hex_chars == 0 or line_hex_chars % 2:
        raise ValueError("row width must contain an even number of hex characters")
    if any(len(row) != line_hex_chars for row in rows):
        raise ValueError("all rows must have the same width")

    try:
        joined = b"".join(rows).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("input contains non-ASCII characters") from error

    if any(character not in string.hexdigits for character in joined):
        raise ValueError("input contains characters other than 0-9, a-f, or A-F")

    has_lower = any("a" <= character <= "f" for character in joined)
    has_upper = any("A" <= character <= "F" for character in joined)
    if has_lower and has_upper:
        raise ValueError("mixed upper- and lower-case hex is not supported")

    hex_case = "upper" if has_upper else "lower"
    return bytes.fromhex(joined), line_hex_chars, hex_case


def write_part(path, payload, wrap_width):
    wrapped = textwrap.fill(payload, width=wrap_width)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write(wrapped)
        output.write("\n")


def compress_gzip(data):
    candidates = []
    for name, strategy in GZIP_STRATEGIES:
        compressor = zlib.compressobj(
            level=9,
            method=zlib.DEFLATED,
            wbits=31,
            memLevel=9,
            strategy=strategy,
        )
        compressed = compressor.compress(data) + compressor.flush()
        candidates.append((len(compressed), name, compressed))
    _, strategy_name, compressed = min(candidates, key=lambda candidate: candidate[0])
    return compressed, strategy_name


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise SystemExit("error: --chunk-size must be greater than zero")
    if args.wrap_width <= 0:
        raise SystemExit("error: --wrap-width must be greater than zero")

    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"error: input file does not exist: {source}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = source.with_name(f"{source.stem}_transfer")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_data = source.read_bytes()
    try:
        rows, newline_name, final_newline = split_rows(source_data)
        raw_data, line_hex_chars, hex_case = decode_hex_rows(rows)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    compressed, compression_strategy = compress_gzip(raw_data)
    payload = base64.b85encode(compressed).decode("ascii")
    parts = [
        payload[offset:offset + args.chunk_size]
        for offset in range(0, len(payload), args.chunk_size)
    ]

    for index, part in enumerate(parts):
        write_part(output_dir / f"part{index:03d}.b85", part, args.wrap_width)

    manifest = {
        "format": FORMAT_VERSION,
        "source_name": source.name,
        "source_size": len(source_data),
        "source_sha256": hashlib.sha256(source_data).hexdigest(),
        "raw_size": len(raw_data),
        "compressed_size": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "payload_chars": len(payload),
        "chunk_size": args.chunk_size,
        "part_count": len(parts),
        "line_count": len(rows),
        "line_hex_chars": line_hex_chars,
        "newline": newline_name,
        "final_newline": final_newline,
        "hex_case": hex_case,
        "compression": "gzip",
        "compression_strategy": compression_strategy,
        "encoding": "base85",
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )

    print(f"source      : {source}")
    print(f"source size : {len(source_data):,} bytes")
    print(f"raw size    : {len(raw_data):,} bytes")
    print(f"compressed  : {len(compressed):,} bytes")
    print(f"strategy    : {compression_strategy}")
    print(f"Base85      : {len(payload):,} characters")
    print(f"parts       : {len(parts)} x up to {args.chunk_size:,} characters")
    print(f"output      : {output_dir}")
    print(f"SHA-256     : {manifest['source_sha256']}")


if __name__ == "__main__":
    main()
