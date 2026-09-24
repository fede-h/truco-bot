#!/usr/bin/env python3
"""Model Exporter for Truco CFR Models.

Exports raw CFR training checkpoints (.bin with regrets + strategy_sums) into
compact, high-speed inference formats for production deployment.

Formats supported:
  - compact (41 bytes/node): u64 key + u8 mask + 8x f32 probabilities
  - quantized (17 bytes/node): u64 key + u8 mask + 8x u8 probabilities (p * 255)
  - compressed (.gz / .zst): applies fast stream compression
"""

import argparse
import gzip
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco_bot.agents.cfr.native_agent import NativeCFRAgent

MAGIC_HEADER = b"TCFR"
FORMAT_FLOAT32 = 0
FORMAT_QUANTIZED_U8 = 1
ENTRY_SIZE_RAW = 76  # 8 (key) + 4 (mask) + 32 (regrets) + 32 (strategy_sum)


def compute_average_strategy_from_raw(
    mask: int,
    regrets: tuple[float, ...],
    strat_sum: tuple[float, ...],
) -> list[float]:
    """Compute normalized strategy distribution for legal actions."""
    out = [0.0] * 8
    sum_strat = 0.0

    for a in range(8):
        if mask & (1 << a):
            val = max(0.0, strat_sum[a])
            out[a] = val
            sum_strat += val

    if sum_strat > 1e-6:
        inv = 1.0 / sum_strat
        for a in range(8):
            if mask & (1 << a):
                out[a] *= inv
        return out

    # Fallback to regret matching if strategy sum is zero
    sum_pos_regret = 0.0
    for a in range(8):
        if mask & (1 << a):
            val = max(0.0, regrets[a])
            out[a] = val
            sum_pos_regret += val

    if sum_pos_regret > 1e-6:
        inv = 1.0 / sum_pos_regret
        for a in range(8):
            if mask & (1 << a):
                out[a] *= inv
        return out

    # Uniform over legal actions
    count = bin(mask).count("1")
    if count > 0:
        uniform = 1.0 / count
        for a in range(8):
            if mask & (1 << a):
                out[a] = uniform

    return out


def export_checkpoint(
    input_path: str | Path,
    output_dir: str | Path = "models/exports",
    quantize: bool = False,
    compress: bool = True,
) -> dict:
    """Export raw checkpoint into production inference format with metadata card."""
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = in_path.stem
    raw_size = in_path.stat().st_size
    num_nodes = raw_size // ENTRY_SIZE_RAW

    fmt_suffix = "q8" if quantize else "f32"
    comp_suffix = ".gz" if compress else ""
    out_filename = f"{model_name}.{fmt_suffix}.bin{comp_suffix}"
    out_path = out_dir / out_filename

    print("=" * 75)
    print(f"EXPORTING CFR MODEL: {model_name}")
    print(f"Source:           {in_path} ({raw_size / (1024*1024):.1f} MB, {num_nodes:,} nodes)")
    print(f"Target format:    {'Quantized uint8 (17 B/node)' if quantize else 'Normalized float32 (41 B/node)'}")
    print(f"Compression:      {'Gzip stream' if compress else 'Uncompressed'}")
    print(f"Target output:    {out_path}")
    print("=" * 75)

    t0 = time.perf_counter()
    hasher = hashlib.sha256()

    open_fn = gzip.open if compress else open
    with open(in_path, "rb") as fin, open_fn(out_path, "wb") as fout:
        # Write 16-byte header: MAGIC (4B) + VERSION (2B) + FORMAT (2B) + NODE_COUNT (8B)
        fmt_id = FORMAT_QUANTIZED_U8 if quantize else FORMAT_FLOAT32
        header = struct.pack("<4sHHQ", MAGIC_HEADER, 1, fmt_id, num_nodes)
        fout.write(header)

        chunk_size = 50_000 * ENTRY_SIZE_RAW
        nodes_processed = 0

        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            num_in_chunk = len(chunk) // ENTRY_SIZE_RAW
            batch_bytes = bytearray()

            for i in range(num_in_chunk):
                offset = i * ENTRY_SIZE_RAW
                # Unpack: Q (u64 key), I (u32 mask), 8f (regrets), 8f (strat_sum)
                key, mask, *floats = struct.unpack_from("<QI8f8f", chunk, offset)
                regrets = floats[:8]
                strat_sum = floats[8:]

                probs = compute_average_strategy_from_raw(mask, regrets, strat_sum)
                legal_mask_u8 = mask & 0xFF

                if quantize:
                    # Quantize probabilities to uint8 (0..255)
                    q_probs = [min(255, max(0, int(round(p * 255)))) for p in probs]
                    batch_bytes.extend(struct.pack("<QB8B", key, legal_mask_u8, *q_probs))
                else:
                    batch_bytes.extend(struct.pack("<QB8f", key, legal_mask_u8, *probs))

                nodes_processed += 1

            fout.write(batch_bytes)

    export_duration = time.perf_counter() - t0
    exported_size = out_path.stat().st_size
    reduction = (1.0 - (exported_size / raw_size)) * 100

    # Compute SHA256 of export
    with open(out_path, "rb") as f:
        while b := f.read(65536):
            hasher.update(b)
    sha256_hex = hasher.hexdigest()

    manifest = {
        "model_name": model_name,
        "source_checkpoint": str(in_path),
        "source_size_bytes": raw_size,
        "source_nodes": num_nodes,
        "exported_file": str(out_path),
        "exported_size_bytes": exported_size,
        "compression_reduction_pct": round(reduction, 2),
        "format": "quantized_uint8" if quantize else "normalized_float32",
        "compressed": compress,
        "sha256": sha256_hex,
        "export_time_sec": round(export_duration, 2),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    manifest_path = out_dir / f"{model_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Export complete in {export_duration:.2f}s!")
    print(f"  Export size:    {exported_size / (1024*1024):.2f} MB ({reduction:.1f}% size reduction)")
    print(f"  SHA-256:        {sha256_hex[:16]}...")
    print(f"  Manifest:       {manifest_path}")
    print("=" * 75 + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CFR model checkpoints into optimized production formats")
    parser.add_argument("--model", type=str, required=True, help="Path to input .bin checkpoint")
    parser.add_argument("--output-dir", type=str, default="models/exports", help="Output directory for exports")
    parser.add_argument("--quantize", action="store_true", help="Quantize probabilities to uint8 (17 bytes/node)")
    parser.add_argument("--no-compress", action="store_true", help="Disable gzip compression")
    args = parser.parse_args()

    export_checkpoint(
        input_path=args.model,
        output_dir=args.output_dir,
        quantize=args.quantize,
        compress=not args.no_compress,
    )


if __name__ == "__main__":
    main()
