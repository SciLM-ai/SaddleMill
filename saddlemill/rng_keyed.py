"""Deterministic attempt-keyed random-number architecture for SaddleMill.

``attempt_keyed_v1`` derives every named stochastic substream from explicit
logical identities using canonical JSON + SHA-256.  No Python ``hash()`` value,
worker/rank identity, file ordering, or process-global stream position enters the
key.

The legacy global-stream behavior remains available as ``legacy_stream`` and is
intentionally implemented outside this module by the historical Dimer seeding
path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import random
import threading
from typing import Any, Iterator, Mapping

import numpy as np

RNG_SCHEME_LEGACY = "legacy_stream"
RNG_SCHEME_ATTEMPT_KEYED_V1 = "attempt_keyed_v1"
RNG_KEY_SCHEMA = "saddlemill_attempt_rng_key_v1"
RNG_GROUP_KEY_SCHEMA = "saddlemill_rng_group_key_v1"

_GLOBAL_RNG_LOCK = threading.RLock()


def _canonical_scalar(value: Any, *, field_name: str) -> str | int | float | bool | None:
    """Return a JSON-stable scalar for a seed identity field.

    Seed identities are deliberately restricted to scalars so accidental object
    reprs, dictionary ordering, NumPy scalar encodings, or path-like object
    implementations cannot silently enter the scientific seed definition.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite; got {value!r}")
        return value
    raise TypeError(
        f"{field_name} must be a JSON scalar (str/int/float/bool/None); "
        f"got {type(value).__name__}"
    )


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Canonical UTF-8 JSON serialization used by the v1 seed contract."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(canonical_key: str) -> str:
    return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()


def _seed_int_from_digest(digest_hex: str) -> int:
    # 128 deterministic bits are ample for local Generator/Random construction.
    return int(digest_hex[:32], 16)


def _legacy_numpy_seed_from_digest(digest_hex: str) -> int:
    # np.random.seed() is the legacy MT19937 compatibility API and accepts uint32.
    return int(digest_hex[32:40], 16)


def _find_info_value(info: Mapping[str, Any] | None, key: str) -> Any:
    """Search top-level-first through nested ``orig_info`` dictionaries."""
    current = info
    seen: set[int] = set()
    while isinstance(current, Mapping) and id(current) not in seen:
        seen.add(id(current))
        value = current.get(key)
        if value not in (None, ""):
            return value
        current = current.get("orig_info")
    return None


def resolve_structure_seed_identity(atoms, config_dict: Mapping[str, Any]) -> str:
    """Resolve the explicit stable structure identity for keyed RNG.

    Precedence is input-frame metadata ``rng_structure_identity`` (top-level,
    then nested ``orig_info``) followed by ``[Main] rng_structure_identity``.
    There is deliberately no fallback to src_index, path name, rank, or file
    ordering because those are exactly the unstable execution identities this
    scheme is designed to remove.
    """
    value = _find_info_value(getattr(atoms, "info", None), "rng_structure_identity")
    if value in (None, ""):
        value = (config_dict.get("Main", {}) or {}).get("rng_structure_identity")
    if value in (None, ""):
        raise ValueError(
            "rng_scheme=attempt_keyed_v1 requires a stable structure seed identity. "
            "Set atoms.info['rng_structure_identity'] (recommended for multi-structure "
            "inputs) or [Main] rng_structure_identity for a single-structure run."
        )
    return str(value)


@dataclass
class _NamedSubstream:
    name: str
    canonical_key: str
    digest: str
    seed_int: int
    legacy_numpy_seed: int
    _py: random.Random | None = None
    _np: np.random.Generator | None = None
    _global_py_state: object | None = None
    _global_np_state: tuple | None = None

    def python(self) -> random.Random:
        if self._py is None:
            self._py = random.Random(self.seed_int)
        return self._py

    def numpy(self) -> np.random.Generator:
        if self._np is None:
            # Freeze the v1 bit-generator contract explicitly; do not inherit a
            # future NumPy change to default_rng()'s default bit generator.
            self._np = np.random.Generator(np.random.PCG64(self.seed_int))
        return self._np

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_key": self.canonical_key,
            "sha256": self.digest,
            "seed_int_128": str(self.seed_int),
            "legacy_numpy_seed_uint32": int(self.legacy_numpy_seed),
        }


@dataclass
class AttemptKeyedRNG:
    """One logical attempt with independent cached named substreams."""

    base_payload: dict[str, Any]
    canonical_attempt_key: str
    attempt_digest: str
    _substreams: dict[str, _NamedSubstream] = field(default_factory=dict)

    @property
    def scheme(self) -> str:
        return RNG_SCHEME_ATTEMPT_KEYED_V1

    def _substream(self, name: str) -> _NamedSubstream:
        name = str(name).strip()
        if not name:
            raise ValueError("RNG substream name must be non-empty")
        stream = self._substreams.get(name)
        if stream is None:
            payload = dict(self.base_payload)
            payload["substream"] = name
            key = canonical_json(payload)
            digest = _digest(key)
            stream = _NamedSubstream(
                name=name,
                canonical_key=key,
                digest=digest,
                seed_int=_seed_int_from_digest(digest),
                legacy_numpy_seed=_legacy_numpy_seed_from_digest(digest),
            )
            self._substreams[name] = stream
        return stream

    def python(self, name: str) -> random.Random:
        return self._substream(name).python()

    def numpy(self, name: str) -> np.random.Generator:
        return self._substream(name).numpy()

    def substream_provenance(self, name: str) -> dict[str, Any]:
        return self._substream(name).provenance()

    @contextmanager
    def global_compatibility(self, name: str) -> Iterator[None]:
        """Temporarily install a deterministic legacy global RNG state.

        This exists only for external APIs that cannot accept an RNG object.
        Previous Python and NumPy module-global states are restored exactly on
        exit, including exceptions.  The lock prevents two SaddleMill keyed
        compatibility regions in the same process from interleaving.
        """
        stream = self._substream(name)
        with _GLOBAL_RNG_LOCK:
            py_state = random.getstate()
            np_state = np.random.get_state()
            try:
                if stream._global_py_state is None:
                    random.seed(stream.seed_int)
                else:
                    random.setstate(stream._global_py_state)
                if stream._global_np_state is None:
                    np.random.seed(stream.legacy_numpy_seed)
                else:
                    np.random.set_state(stream._global_np_state)
                yield
            finally:
                # Preserve advancement inside this named external substream while
                # restoring the unrelated process-global states exactly.
                stream._global_py_state = random.getstate()
                stream._global_np_state = np.random.get_state()
                random.setstate(py_state)
                np.random.set_state(np_state)

    def provenance(self) -> dict[str, Any]:
        return {
            "rng_scheme": RNG_SCHEME_ATTEMPT_KEYED_V1,
            "rng_key_schema": RNG_KEY_SCHEMA,
            "base_seed": self.base_payload["base_seed"],
            "seed_group_identity": self.base_payload["seed_group_identity"],
            "structure_seed_identity": self.base_payload["structure_seed_identity"],
            "logical_attempt_identity": dict(self.base_payload["logical_attempt_identity"]),
            "canonical_attempt_key": self.canonical_attempt_key,
            "attempt_key_sha256": self.attempt_digest,
            "substreams": {
                name: stream.provenance()
                for name, stream in sorted(self._substreams.items())
            },
        }


@dataclass
class StructureRNGFactory:
    """Factory for attempt and structure/reaction-group deterministic RNG keys."""

    base_seed: str | int | float | bool | None
    seed_group_identity: str
    structure_seed_identity: str
    _attempts: dict[tuple[int, str], AttemptKeyedRNG] = field(default_factory=dict)
    _groups: dict[tuple[str, str], AttemptKeyedRNG] = field(default_factory=dict)

    def _build(self, logical_attempt_identity: Mapping[str, Any], *, schema: str) -> AttemptKeyedRNG:
        payload = {
            "rng_key_schema": schema,
            "rng_scheme": RNG_SCHEME_ATTEMPT_KEYED_V1,
            "base_seed": _canonical_scalar(self.base_seed, field_name="rng_base_seed"),
            "seed_group_identity": str(self.seed_group_identity),
            "structure_seed_identity": str(self.structure_seed_identity),
            "logical_attempt_identity": dict(logical_attempt_identity),
        }
        key = canonical_json(payload)
        return AttemptKeyedRNG(
            base_payload=payload,
            canonical_attempt_key=key,
            attempt_digest=_digest(key),
        )

    def for_attempt(self, attempt_id: int, configured_reaction_type: str) -> AttemptKeyedRNG:
        token = (int(attempt_id), str(configured_reaction_type))
        rng = self._attempts.get(token)
        if rng is None:
            rng = self._build(
                {
                    "attempt_id": int(attempt_id),
                    "configured_reaction_type": str(configured_reaction_type),
                },
                schema=RNG_KEY_SCHEMA,
            )
            self._attempts[token] = rng
        return rng

    def for_group(self, configured_reaction_type: str, group_name: str) -> AttemptKeyedRNG:
        """Stable group key for algorithms requiring coupled without-replacement assignment.

        The group key is used only where the pre-existing candidate-generation
        algorithm intentionally couples slots (for example a random permutation
        sampled without replacement).  It is independent of execution order and
        all per-attempt stochastic decisions remain in attempt keys.
        """
        token = (str(configured_reaction_type), str(group_name))
        rng = self._groups.get(token)
        if rng is None:
            rng = self._build(
                {
                    "attempt_id": "__group__",
                    "configured_reaction_type": str(configured_reaction_type),
                    "group_name": str(group_name),
                },
                schema=RNG_GROUP_KEY_SCHEMA,
            )
            self._groups[token] = rng
        return rng


    def provenance_for_attempt(self, attempt_id: int, configured_reaction_type: str) -> dict[str, Any]:
        provenance = self.for_attempt(attempt_id, configured_reaction_type).provenance()
        group_keys = {}
        for (rtype, group_name), group_rng in sorted(self._groups.items()):
            if rtype == str(configured_reaction_type):
                group_keys[group_name] = group_rng.provenance()
        if group_keys:
            provenance["group_seed_keys"] = group_keys
        return provenance


def build_structure_rng_factory(config_dict: Mapping[str, Any], atoms) -> StructureRNGFactory | None:
    main = config_dict.get("Main", {}) or {}
    scheme = str(main.get("rng_scheme", RNG_SCHEME_LEGACY)).strip().lower()
    if scheme == RNG_SCHEME_LEGACY:
        return None
    if scheme != RNG_SCHEME_ATTEMPT_KEYED_V1:
        raise ValueError(
            f"[Main] rng_scheme must be {RNG_SCHEME_LEGACY!r} or "
            f"{RNG_SCHEME_ATTEMPT_KEYED_V1!r}; got {scheme!r}"
        )
    structure_identity = resolve_structure_seed_identity(atoms, config_dict)
    return StructureRNGFactory(
        base_seed=main.get("rng_base_seed", 0),
        seed_group_identity=str(main.get("rng_seed_group", "default")),
        structure_seed_identity=structure_identity,
    )


def seed_key_diagnostic(
    *,
    base_seed: Any,
    seed_group_identity: str,
    structure_seed_identity: str,
    attempt_id: int,
    configured_reaction_type: str,
    substreams: list[str],
) -> dict[str, Any]:
    factory = StructureRNGFactory(
        base_seed=base_seed,
        seed_group_identity=seed_group_identity,
        structure_seed_identity=structure_seed_identity,
    )
    rng = factory.for_attempt(attempt_id, configured_reaction_type)
    for name in substreams:
        rng.substream_provenance(name)
    return rng.provenance()




def _parse_cli_scalar(text: str) -> str | int | float | bool | None:
    """Parse a CLI seed scalar using JSON literals when possible.

    This makes ``--base-seed 42`` match an INI value parsed as integer 42 while
    still allowing ordinary unquoted string seeds such as ``campaign-A``.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if value is None or isinstance(value, (str, int, float, bool)):
        return _canonical_scalar(value, field_name="rng_base_seed")
    raise ValueError("--base-seed must decode to a JSON scalar, not a list/object")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the canonical SaddleMill attempt_keyed_v1 seed key and substream seeds."
    )
    parser.add_argument("--base-seed", required=True)
    parser.add_argument("--seed-group", required=True)
    parser.add_argument("--structure-id", required=True)
    parser.add_argument("--attempt-id", required=True, type=int)
    parser.add_argument("--reaction-type", required=True)
    parser.add_argument(
        "--substream",
        action="append",
        default=[],
        help="Named stochastic substream; repeat for multiple streams.",
    )
    args = parser.parse_args()
    payload = seed_key_diagnostic(
        base_seed=_parse_cli_scalar(args.base_seed),
        seed_group_identity=args.seed_group,
        structure_seed_identity=args.structure_id,
        attempt_id=args.attempt_id,
        configured_reaction_type=args.reaction_type,
        substreams=args.substream,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
