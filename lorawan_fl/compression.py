"""Gradient sparsification, quantization, and binary update encoding."""

from __future__ import annotations

import struct
import zlib
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .config import CompressionConfig, HEADER_BASE, HEADER_QUANT_EXTRA, INDEX_BYTES
from .update_types import ReconstructedUpdate

# Compatibility name for EXP008-era callers.  New code should use the more
# explicit ReconstructedUpdate(values, valid_mask).
FragmentMaskedUpdate = ReconstructedUpdate

def _pack_4bit(arr):
    arr    = arr.astype(np.uint8)
    packed = np.zeros((arr.size + 1) // 2, dtype=np.uint8)
    for i in range(arr.size):
        if i % 2 == 0: packed[i//2] |= arr[i] & 0xF
        else:           packed[i//2] |= (arr[i] & 0xF) << 4
    return packed

def _unpack_4bit(packed, shape):
    n = int(np.prod(shape)); out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i % 2 == 0: out[i] = packed[i//2] & 0xF
        else:           out[i] = (packed[i//2] >> 4) & 0xF
    return out.reshape(shape)

def _pack_2bit(arr):
    arr    = arr.astype(np.uint8)
    packed = np.zeros((arr.size + 3) // 4, dtype=np.uint8)
    for i in range(arr.size):
        packed[i//4] |= (arr[i] & 0x3) << ((i % 4) * 2)
    return packed

def _unpack_2bit(packed, shape):
    n = int(np.prod(shape)); out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = (packed[i//4] >> ((i % 4) * 2)) & 0x3
    return out.reshape(shape)

def _pack_1bit(arr):
    arr    = arr.astype(np.uint8)
    packed = np.zeros((arr.size + 7) // 8, dtype=np.uint8)
    for i in range(arr.size):
        packed[i//8] |= (arr[i] & 0x1) << (i % 8)
    return packed

def _unpack_1bit(packed, shape):
    n = int(np.prod(shape)); out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = (packed[i//8] >> (i % 8)) & 0x1
    return out.reshape(shape)

def _bytes_per_entry(q_bits):
    return INDEX_BYTES + (q_bits / 8 if q_bits in (1, 2, 4) else 4.0)

def _header_bytes(q_bits):
    return HEADER_BASE + (HEADER_QUANT_EXTRA if q_bits in (1, 2, 4) else 0)

def budget_k(max_payload_bytes, q_bits):
    available = max_payload_bytes - _header_bytes(q_bits)
    if available <= 0:
        return 1
    return max(1, int(available / _bytes_per_entry(q_bits)))

def _quantize_values(vals, q_bits):
    if q_bits in (1, 2, 4):
        mn, mx = float(vals.min()), float(vals.max())
        levels = (2 ** q_bits) - 1
        q = (np.zeros(len(vals), dtype=np.uint8) if mx == mn
             else np.round((vals - mn) / (mx - mn) * levels).astype(np.uint8))
        if   q_bits == 4: packed = _pack_4bit(q)
        elif q_bits == 2: packed = _pack_2bit(q)
        else:             packed = _pack_1bit(q)
        return packed.tobytes(), (mn, mx)
    return vals.astype(np.float32).tobytes(), None

def _dequantize_values(data, k, q_bits, min_max):
    if q_bits in (1, 2, 4):
        packed = np.frombuffer(data, dtype=np.uint8)
        if   q_bits == 4: q = _unpack_4bit(packed, (k,)).astype(np.float32)
        elif q_bits == 2: q = _unpack_2bit(packed, (k,)).astype(np.float32)
        else:             q = _unpack_1bit(packed, (k,)).astype(np.float32)
        mn, mx = min_max
        return q / ((2 ** q_bits) - 1) * (mx - mn) + mn
    return np.frombuffer(data, dtype=np.float32).copy()

def _encode_sparse(indices, val_bytes, N, k, q_bits, min_max):
    hdr = struct.pack("<BIH", q_bits, N, k)
    if min_max is not None:
        hdr += struct.pack("<ff", min_max[0], min_max[1])
    idx_bytes = b"".join(
        struct.pack("<I", int(i) & 0xFFFFFF)[:INDEX_BYTES] for i in indices)
    return hdr + idx_bytes + val_bytes

def _decode_sparse(blob):
    q_bits, N, k = struct.unpack_from("<BIH", blob, 0)
    offset = HEADER_BASE; min_max = None
    if q_bits in (1, 2, 4):
        mn, mx   = struct.unpack_from("<ff", blob, offset)
        min_max  = (mn, mx); offset += HEADER_QUANT_EXTRA
    idx_end = offset + INDEX_BYTES * k
    idx_raw = blob[offset:idx_end]
    if len(idx_raw) < INDEX_BYTES * k:
        raise ValueError(f"Short index block: expected {INDEX_BYTES*k}B, got {len(idx_raw)}B")
    indices = np.array([
        struct.unpack("<I", idx_raw[i*INDEX_BYTES:(i+1)*INDEX_BYTES] + b"\x00")[0]
        for i in range(k)], dtype=np.int64)
    if len(indices) > 0 and np.max(indices) >= N:
        raise ValueError(f"Index {np.max(indices)} out of bounds for N={N}")
    values = _dequantize_values(blob[idx_end:], k, q_bits, min_max)
    return indices, values, N

def flatten_update(model, reference_model) -> np.ndarray:
    """Return one dense CPU parameter update in model-parameter order."""
    ref_params = (dict(reference_model.named_parameters())
                  if reference_model is not None else None)
    parts = []
    for name, param in model.named_parameters():
        p = param.data.cpu().float()
        if ref_params is not None:
            p = p - ref_params[name].data.cpu().float()
        parts.append(p.flatten())
    return torch.cat(parts).numpy()


def _compress_flat_update(flat: np.ndarray, cfg: CompressionConfig):
    N    = len(flat)
    q    = cfg.quantization_bits if cfg.use_quantization else 0
    k    = min(budget_k(cfg.max_payload_bytes, q), N)
    if cfg.use_sparsity and k < N:
        top_idx = np.argpartition(np.abs(flat), -k)[-k:]
        top_idx = top_idx[np.argsort(np.abs(flat[top_idx]))[::-1]]
    else:
        top_idx = np.arange(N, dtype=np.int64); k = N
    top_vals        = flat[top_idx]
    val_bytes, mm   = _quantize_values(top_vals, q)
    blob            = _encode_sparse(top_idx, val_bytes, N, k, q, mm)
    if cfg.apply_zlib:
        blob = zlib.compress(blob, level=9)
    return blob, len(blob), k, N, top_idx


def compress_update(model, reference_model, cfg: CompressionConfig):
    flat = flatten_update(model, reference_model)
    blob, blob_bytes, k, N, _ = _compress_flat_update(flat, cfg)
    return blob, blob_bytes, k, N


def compress_update_with_information(model, reference_model, cfg: CompressionConfig):
    """
    Compress an update and retain its dense pre-compression representation.

    The retained-energy ratio is measured before quantization, over the actual
    top-k coordinates selected by the payload budget.
    """
    flat = flatten_update(model, reference_model)
    return compress_flat_update_with_information(flat, cfg)


def compress_flat_update_with_information(flat_update, cfg: CompressionConfig):
    """Compress an already materialized dense update and retain diagnostics."""
    if isinstance(flat_update, torch.Tensor):
        flat = flat_update.detach().cpu().float().numpy()
    else:
        flat = np.asarray(flat_update, dtype=np.float32)
    blob, blob_bytes, k, N, top_idx = _compress_flat_update(flat, cfg)
    full_energy = float(np.dot(flat, flat))
    retained_energy = float(np.dot(flat[top_idx], flat[top_idx]))
    retained_energy_ratio = (
        retained_energy / full_energy if full_energy > 0.0 else 1.0
    )
    return (
        blob,
        blob_bytes,
        k,
        N,
        torch.from_numpy(flat.copy()),
        retained_energy_ratio,
    )

def encoded_update_payload_bytes(model: nn.Module, cfg: CompressionConfig) -> int:
    """
    Return the encoded sparse-update size implied by the compression budget.

    This is used by the communication pre-pass, where the transmitter needs an
    opaque payload size before the expensive local training step is performed.
    """
    N = sum(param.numel() for param in model.parameters())
    q = cfg.quantization_bits if cfg.use_quantization else 0
    k = min(budget_k(cfg.max_payload_bytes, q), N)
    if not cfg.use_sparsity:
        k = N

    if q == 4:
        value_bytes = (k + 1) // 2
    elif q == 2:
        value_bytes = (k + 3) // 4
    elif q == 1:
        value_bytes = (k + 7) // 8
    else:
        value_bytes = 4 * k

    raw_bytes = _header_bytes(q) + INDEX_BYTES * k + value_bytes
    if cfg.apply_zlib:
        # The exact compressed length depends on the trained values. Use the
        # uncompressed upper bound to avoid underestimating communication cost.
        return min(raw_bytes, cfg.max_payload_bytes)
    return raw_bytes

def decompress_update(blob: bytes, apply_zlib: bool) -> torch.Tensor:
    raw = zlib.decompress(blob) if apply_zlib else blob
    indices, values, N = _decode_sparse(raw)
    dense = np.zeros(N, dtype=np.float32)
    dense[indices] = values
    return torch.tensor(dense, dtype=torch.float32)


def decode_sparse_update(blob: bytes, apply_zlib: bool):
    """Return encoded coordinate indices, values, and dense dimension."""

    raw = zlib.decompress(blob) if apply_zlib else blob
    return _decode_sparse(raw)


def decompress_update_with_fragment_mask(
    blob: bytes,
    apply_zlib: bool,
    delivered_fragment_indices,
    num_fragments: int,
) -> torch.Tensor:
    """Decode an update and zero coordinates assigned to lost fragments.

    Sparse coordinate/value records are sorted by model-coordinate index and
    divided as evenly as possible into independently decodable fragments. A
    delivered fragment contributes its records; records in collided fragments
    are reconstructed as zero. The dense update dimension is unchanged.

    This coordinate-shard abstraction avoids zeroing arbitrary byte ranges in
    the legacy monolithic blob, which could corrupt shared headers or indices
    and turn an erasure into a decoder-format failure.
    """
    return decompress_update_with_fragment_observation(
        blob,
        apply_zlib,
        delivered_fragment_indices,
        num_fragments,
    ).values


def decompress_update_with_fragment_observation(
    blob: bytes,
    apply_zlib: bool,
    delivered_fragment_indices,
    num_fragments: int,
) -> FragmentMaskedUpdate:
    """Decode a partial update and identify collision-erased coordinates.

    ``observed`` is one for coordinates whose compressed value is known and
    zero only for sparse records assigned to collided fragments. Coordinates
    deliberately omitted by top-k compression remain observed zeros; this
    keeps collision-erasure normalization distinct from nonzero-only averaging.
    """
    # Local import avoids a compression↔fragmentation import cycle while
    # preserving the historical public helper.
    from .fragmentation import (
        receptions_from_delivered_indices,
        reconstruct_partial_update,
    )

    receptions = receptions_from_delivered_indices(
        client_id=0,
        round_index=0,
        total_fragments=num_fragments,
        delivered_fragment_indices=delivered_fragment_indices,
    )
    return reconstruct_partial_update(blob, apply_zlib, receptions)

def split_flat_gradient(flat: torch.Tensor, model: nn.Module) -> Dict:
    out = {}; offset = 0
    for name, param in model.named_parameters():
        n = param.numel()
        out[name] = flat[offset:offset + n].view(param.shape)
        offset += n
    return out

def aggregate_updates(flat_tensors: List[torch.Tensor]) -> torch.Tensor:
    if not flat_tensors:
        return torch.zeros(0)
    return torch.stack(flat_tensors).mean(dim=0)

# ---------------------------------------------------------------------------
