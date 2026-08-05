"""Flat (container-free) serialization for one ``transform()``-shaped sample.

Why this exists: ``torch.save`` writes a **zip archive with one record per
tensor**, and a transformed sample is ~600 tiny tensors (one per card slot per
flag, across a 60-deep decision chain). Measured on the pretraining cache, a
sample whose tensors hold **21 KB of actual numbers** serialized to a **269 KB**
blob — 12.8x, all of it zip record headers, the central directory, and per-record
CRCs — and ``torch.load`` spent **26 ms** per sample parsing that archive. At
batch 64 that is 1.7 s of pure CPU per batch against a 115 ms GPU step, i.e. the
GPU idling ~80% of the time no matter how many loader workers you throw at it.
That cost is the *container*, not the data, and precomputing features does not
touch it — which is why training stayed CPU-bound after the precompute.

This module stores the same tensors with no container at all:

- The **structure** (nested dict/list shape, key order, per-leaf dtype and rank)
  is identical across samples except for the two variable-length lists
  ``state.bench_pokemon`` and ``opponent_state.bench_pokemon`` (0..5 each). It is
  therefore hoisted out of the samples entirely and stored **once per distinct
  shape** in the manifest, as a *skeleton* (see ``skeleton``). A sample's record
  only carries an index into that table.
- The **payload** is raw tensor bytes, concatenated, grouped by dtype in
  descending element size (``_BLOCK_ORDER``). Grouping is what makes every block
  naturally aligned for ``torch.frombuffer`` without a single padding byte
  between tensors — a per-tensor alignment pad would have cost more than the
  shape header does.
- The only per-sample variability left is the extents, stored as a flat
  ``uint16`` array in the record header.

- That body is then **zlib-compressed** (``COMPRESSION_LEVEL``), which on real
  records is a further **52x** — the payload is mostly structural zeros. This is
  what takes the full cache from ~130GB to ~2.5GB, i.e. from "randomly read off
  disk every epoch" to "resident in the page cache".

Decoding is one ``zlib.decompress`` plus ``torch.frombuffer`` views over the
resulting buffer and a walk of the cached skeleton: no unpickling, no archive
parsing, no per-tensor copies.

The tensors handed back **alias that buffer**, so the caller must not mutate them
in place. Nothing does today: ``collate.pad_stack`` allocates its own output and
copies into it, which is every field's only path into a batch.

Values are bit-identical to what ``torch.save`` stored — this changes the
encoding, never the data, so a cache repacked into this format trains to exactly
the same numbers as the pickled one it came from.
"""

import json
import struct
import zlib
from array import array

import torch

#: zlib level for a record's body. Measured on real records: **52x** smaller at
#: level 6 (34x at level 1), decompressing in 0.25ms — against the 2.27ms the
#: rest of ``decode`` costs, so compression is close to free on the read side
#: and the level barely moves that number. What the 52x buys is not disk: it is
#: that the whole 668k-sample dataset lands at ~2.5GB instead of ~130GB, small
#: enough for the OS page cache to hold it, so shuffled reads stop hitting the
#: disk at all. The data compresses this absurdly well because it is mostly
#: structural zeros — padded card slots, a 60-deep decision chain whose options
#: repeat, and one all-zero float field that alone is a sixth of the payload.
#:
#: Level 6 over 1: writing is 141MB/s vs 346MB/s per core, and the repack is
#: bottlenecked on *reading* the old format (26ms/sample) by a wide margin, so
#: the slower setting costs nothing there either.
COMPRESSION_LEVEL = 6

#: dtype <-> small int, so a skeleton leaf is two integers rather than a string.
#: Append-only: the integers are baked into every written cache, so reordering
#: this list silently reinterprets existing data. Anything not listed here
#: raises at encode time rather than being coerced.
_DTYPES = (
    torch.int64,
    torch.float64,
    torch.float32,
    torch.int32,
    torch.int16,
    torch.bool,
    torch.uint8,
)
_CODE = {dtype: code for code, dtype in enumerate(_DTYPES)}

#: Payload block order: **descending element size**. This is load-bearing, not
#: cosmetic. Blocks laid out largest-element-first mean every block starts at an
#: offset that is already a multiple of its own element size (the int64 block
#: starts 8-aligned and has 8-byte elements, so the float32 block after it starts
#: 4-aligned, and so on), which is exactly what ``torch.frombuffer`` requires.
#: Interleaving dtypes instead would need up to 7 pad bytes *per tensor* — ~4 KB
#: on a 21 KB sample.
_BLOCK_ORDER = tuple(
    sorted(range(len(_DTYPES)), key=lambda code: -_DTYPES[code].itemsize)
)

#: ``num_target``, ``episode_id``, ``frame_index``, ``player_index``,
#: ``len(player_name)``. '<' (little-endian, packed) rather than native so a
#: cache is not silently tied to the machine that wrote it.
#:
#: The template index is deliberately *not* in here: it is assigned by the
#: writer's parent process, after the worker has already compressed the body,
#: so it has to live outside the compressed region (see ``pack``).
_HEADER = struct.Struct("<IqqqH")

#: The template index, stored uncompressed ahead of each record's body.
_SIGNATURE = struct.Struct("<I")

#: Marks a tensor slot in a skeleton. A single-key dict rather than a bare list
#: because a *container* list is also a list — the two need to stay
#: distinguishable when the skeleton round-trips through JSON.
_LEAF = "@"


def skeleton(obj, leaves: list[torch.Tensor]):
    """Split ``obj`` into (structure, tensors-in-DFS-order).

    ``leaves`` is appended to in place; the return value is the JSON-safe
    structure with every tensor replaced by ``{"@": [dtype_code, ndim]}``.
    Extents are deliberately *not* part of the skeleton — they are the one thing
    that varies sample to sample, so they live in the per-sample header while
    everything here is shared across every sample of the same shape.
    """
    if isinstance(obj, torch.Tensor):
        leaves.append(obj)
        code = _CODE.get(obj.dtype)
        if code is None:
            raise TypeError(f"dtype {obj.dtype} is not in flat_codec._DTYPES")
        return {_LEAF: [code, obj.dim()]}
    if isinstance(obj, dict):
        return {key: skeleton(value, leaves) for key, value in obj.items()}
    if isinstance(obj, list):
        return [skeleton(value, leaves) for value in obj]
    raise TypeError(f"cannot serialize {type(obj)} in a feature tree")


class Template:
    """A skeleton preprocessed into the flat arrays ``decode`` walks.

    Built once per distinct structure (there are ~36, one per
    ``(own bench count, opponent bench count)`` pair) and reused for every
    sample that shares it, so none of this per-leaf bookkeeping happens per
    ``__getitem__``.
    """

    __slots__ = ("skel", "codes", "ndims", "total_ndim", "blocks")

    def __init__(self, skel):
        self.skel = skel
        self.codes: list[int] = []
        self.ndims: list[int] = []
        self._scan(skel)
        self.total_ndim = sum(self.ndims)
        # Leaf indexes grouped by dtype, in payload order — the exact iteration
        # order ``encode`` wrote the blocks in.
        self.blocks = [
            (code, [i for i, c in enumerate(self.codes) if c == code])
            for code in _BLOCK_ORDER
        ]
        self.blocks = [(code, idx) for code, idx in self.blocks if idx]

    def _scan(self, node) -> None:
        if isinstance(node, dict):
            leaf = node.get(_LEAF)
            if leaf is not None:
                self.codes.append(leaf[0])
                self.ndims.append(leaf[1])
                return
            for value in node.values():
                self._scan(value)
        else:
            for value in node:
                self._scan(value)


def encode(features: dict, meta: dict, target_action: torch.Tensor) -> tuple[str, bytes]:
    """Serialize one sample. Returns ``(signature, compressed_body)``.

    The signature is the skeleton as canonical JSON — used by the writer both as
    the registry key and as the stored template, so the two can never disagree
    about what a record's template index points at. The index itself is not in
    the body: only the writer's parent sees every sample and can assign it, and
    by then the body is already compressed. ``pack`` prepends it.

    Compression happens here, in the worker, which is both where the spare cores
    are and what shrinks the IPC payload back to a few KB per sample.
    """
    leaves: list[torch.Tensor] = []
    skel = skeleton(features, leaves)

    dims = array("H")
    for tensor in leaves:
        for extent in tensor.shape:
            if extent > 0xFFFF:
                raise ValueError(f"extent {extent} exceeds the uint16 shape header")
        dims.extend(tensor.shape)

    name = str(meta.get("player_name") or "").encode()
    target_action = target_action.to(torch.int64).contiguous()
    record = bytearray(
        _HEADER.pack(
            target_action.numel(),
            int(meta["episode_id"]),
            int(meta["frame_index"]),
            int(meta["player_index"]),
            len(name),
        )
    )
    record += name
    record += dims.tobytes()
    record += b"\0" * (-len(record) % 8)  # the int64 block must start 8-aligned

    # target_action leads the int64 block: it is int64 too, and putting it here
    # rather than in the header keeps the header fixed-width per sample.
    record += target_action.numpy().tobytes()
    for code, _ in Template(skel).blocks:
        for tensor in leaves:
            if _CODE[tensor.dtype] == code:
                record += tensor.detach().contiguous().numpy().tobytes()
    return json.dumps(skel), zlib.compress(bytes(record), COMPRESSION_LEVEL)


def pack(sig_index: int, compressed_body: bytes) -> bytes:
    """Final on-disk bytes for one record: template index, then its body."""
    return _SIGNATURE.pack(sig_index) + compressed_body


def unpack(record) -> tuple[int, bytearray]:
    """Inverse of ``pack``: ``(template_index, decompressed_body)``.

    The body comes back as a ``bytearray`` because ``decode`` hands out tensors
    that are views into it, and ``torch.frombuffer`` warns on every call for a
    read-only buffer. ``zlib.decompress`` returns immutable ``bytes``, hence the
    copy — ~20us against the 0.25ms the decompression itself takes.
    """
    sig_index = _SIGNATURE.unpack_from(record, 0)[0]
    return sig_index, bytearray(zlib.decompress(memoryview(record)[_SIGNATURE.size :]))


def decode(buffer: bytearray, template: Template) -> tuple[dict, dict, torch.Tensor]:
    """Inverse of ``encode``: returns ``(features, meta, target_action)``.

    ``buffer`` is a decompressed body from ``unpack``, not a raw on-disk record.
    Every returned tensor is a **view into** it — see the module docstring.
    """
    num_target, episode_id, frame_index, player_index, name_len = _HEADER.unpack_from(
        buffer, 0
    )
    offset = _HEADER.size
    name = bytes(buffer[offset : offset + name_len]).decode()
    offset += name_len

    # .cast('H') reads the extents as uint16 without copying them out.
    dims = memoryview(buffer)[offset : offset + 2 * template.total_ndim].cast("H")
    offset += 2 * template.total_ndim
    offset += -offset % 8

    shapes = []
    cursor = 0
    for ndim in template.ndims:
        shapes.append(tuple(dims[cursor : cursor + ndim]))
        cursor += ndim

    # num_target == 0 is a real label, not a corrupt record: an expert who
    # declined a decision selected no options. frombuffer rejects count=0, so it
    # needs the same empty-tensor path the feature leaves below use.
    if num_target:
        target_action = torch.frombuffer(
            buffer, dtype=torch.int64, count=num_target, offset=offset
        )
        offset += 8 * num_target
    else:
        target_action = torch.empty(0, dtype=torch.int64)

    tensors: list[torch.Tensor | None] = [None] * len(template.codes)
    for code, indexes in template.blocks:
        dtype = _DTYPES[code]
        itemsize = dtype.itemsize
        for i in indexes:
            shape = shapes[i]
            numel = 1
            for extent in shape:
                numel *= extent
            if numel == 0:
                # frombuffer rejects count=0, and an empty tensor has no bytes
                # to point at anyway.
                tensors[i] = torch.empty(shape, dtype=dtype)
                continue
            tensors[i] = torch.frombuffer(
                buffer, dtype=dtype, count=numel, offset=offset
            ).view(shape)
            offset += numel * itemsize

    features = _rebuild(template.skel, iter(tensors))
    meta = {
        "episode_id": episode_id,
        "frame_index": frame_index,
        "player_index": player_index,
        "player_name": name,
    }
    return features, meta, target_action


def _rebuild(node, tensors):
    if isinstance(node, dict):
        if _LEAF in node:
            return next(tensors)
        return {key: _rebuild(value, tensors) for key, value in node.items()}
    return [_rebuild(value, tensors) for value in node]
