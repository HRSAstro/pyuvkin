"""Memory snapshot helpers."""

from pyuvkin import memory


def test_format_bytes():
    assert memory.format_bytes(None) == "?"
    assert memory.format_bytes(500_000) == "1 MB"
    assert "GB" in memory.format_bytes(2_500_000_000)


def test_snapshot_has_process_and_host():
    snap = memory.snapshot()
    assert snap["rss_bytes"] is not None and snap["rss_bytes"] > 0
    # available/total can be None on exotic hosts; on mac/linux they should exist
    assert snap["available_bytes"] is None or snap["available_bytes"] > 0
    assert snap["total_bytes"] is None or snap["total_bytes"] > 0
    text = memory.format_snapshot(snap)
    assert "RSS" in text and "available" in text
