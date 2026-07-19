from shadowtrace.common.ulid import new_ulid, ulid_timestamp_ms


def test_ulid_is_26_chars() -> None:
    assert len(new_ulid()) == 26


def test_ulid_sortable_by_time() -> None:
    a = new_ulid(ts_ms=1000)
    b = new_ulid(ts_ms=2000)
    assert a < b


def test_ulid_timestamp_roundtrip() -> None:
    ulid = new_ulid(ts_ms=1_700_000_000_000)
    assert ulid_timestamp_ms(ulid) == 1_700_000_000_000


def test_ulid_uniqueness() -> None:
    ids = {new_ulid(ts_ms=42) for _ in range(1000)}
    assert len(ids) == 1000
