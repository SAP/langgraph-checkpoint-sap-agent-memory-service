"""Serde round-trip tests for the base64 ↔ blob encoding pipeline."""

from __future__ import annotations

import base64

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

_serde = JsonPlusSerializer()


def _round_trip(obj: object) -> object:
    """Simulate the full wire round-trip: dumps_typed → base64 → base64decode → loads_typed."""
    type_, data = _serde.dumps_typed(obj)
    encoded = base64.b64encode(data).decode()
    decoded_bytes = base64.b64decode(encoded)
    return _serde.loads_typed((type_, decoded_bytes))


def test_msgpack_round_trip():
    obj = {"id": "snap-1", "channel_values": {"messages": []}, "v": 1}
    result = _round_trip(obj)
    assert result == obj


def test_json_round_trip():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer as JsonSerde

    json_serde = JsonSerde()
    obj = {"key": "value with unicode: élève", "num": 42}
    type_, data = json_serde.dumps_typed(obj)
    encoded = base64.b64encode(data).decode()
    result = json_serde.loads_typed((type_, base64.b64decode(encoded)))
    assert result == obj


def test_null_round_trip():
    """None value round-trips without error."""
    result = _round_trip(None)
    assert result is None


def test_bytes_round_trip():
    """Raw bytes identity preserved."""
    obj = b"\x00\x01\x02\xffbinary"
    result = _round_trip(obj)
    assert result == obj


@pytest.mark.skip(reason="gzip compression not yet implemented")
def test_msgpack_gzip_round_trip():
    pass


@pytest.mark.skip(reason="HDL files offload not yet implemented")
def test_msgpack_gzip_hdl_ref_round_trip():
    pass


def test_serialization_type_fits_column():
    """All known composite type tags fit in SERIALIZATION_TYPE String(50)."""
    tags = [
        "msgpack",
        "json",
        "null",
        "bytes",
        "msgpack+gzip",
        "msgpack+gzip+hdl-ref",
    ]
    for tag in tags:
        assert len(tag) <= 50, f"Type tag '{tag}' exceeds String(50) column limit"
