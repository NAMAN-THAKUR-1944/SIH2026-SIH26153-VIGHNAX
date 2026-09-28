"""Upload sniffing and the streaming endpoint used by the dashboard."""

import bz2
import gzip
import json
import os

import pytest

import server

PCAP_HEADER = b"\xd4\xc3\xb2\xa1" + b"\x02\x00\x04\x00" + b"\x00" * 8 + b"\xff\xff\x00\x00\x01\x00\x00\x00"


@pytest.mark.parametrize("wrap", ["plain", "gz", "bz2"])
def test_sniff_detects_pcap_through_compression(tmp_path, wrap):
    data = {"plain": PCAP_HEADER, "gz": gzip.compress(PCAP_HEADER), "bz2": bz2.compress(PCAP_HEADER)}[wrap]
    p = tmp_path / "capture_without_extension"
    p.write_bytes(data)
    assert server.sniff(str(p)) == "pcap"


def test_sniff_detects_flow_records(tmp_path):
    p = tmp_path / "flows"
    p.write_bytes(gzip.compress(b"StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,TotPkts,TotBytes,SrcBytes,Label\n"))
    assert server.sniff(str(p)) == "flows"


@pytest.mark.skipif(not os.path.isfile(os.path.join(server.ROOT, "models", "world_model.pt")), reason="needs trained model")
def test_stream_sample_emits_every_window():
    client = server.app.test_client()
    res = client.post("/api/stream", data={"sample": "ctu13_s43_neris"})
    events = [json.loads(line) for line in res.get_data(as_text=True).splitlines() if line.strip()]
    meta = next(e for e in events if e["type"] == "meta")
    windows = [e for e in events if e["type"] == "window"]
    done = next(e for e in events if e["type"] == "done")
    assert len(windows) == meta["n"] and [w["i"] for w in windows] == list(range(meta["n"]))
    assert all(0.0 <= w["p_any"] <= 1.0 for w in windows)
    assert done["id"] and isinstance(done["incidents"], list)
    detail = client.get(f"/api/explain/{done['id']}/{meta['n'] - 1}").get_json()
    assert detail["features"] and "flagged" in detail

@pytest.mark.skipif(not os.path.isfile(os.path.join(server.ROOT, "models", "world_model.pt")), reason="needs trained model")
def test_explain_while_streaming_only_processed_windows():
    """The dashboard explains windows while the stream runs, never a window the stream has not reached."""
    client = server.app.test_client()
    res = client.post("/api/stream", data={"sample": "cicids2017_sample", "speed": "0"}, buffered=False)
    lines = (line for chunk in res.response for line in (chunk.decode() if isinstance(chunk, bytes) else chunk).splitlines())
    meta = json.loads(next(line for line in lines if '"meta"' in line))
    assert meta["id"]
    ahead = client.get(f"/api/explain/{meta['id']}/{meta['n'] - 1}")
    assert ahead.status_code == 409
    for line in lines:
        ev = json.loads(line)
        if ev["type"] == "window" and ev["i"] == 5:
            break
    detail = client.get(f"/api/explain/{meta['id']}/5").get_json()
    assert detail["index"] == 5 and detail["features"] and "flagged" in detail
    rest = [json.loads(line) for line in lines]
    assert rest[-1]["type"] == "done" and rest[-1]["id"] == meta["id"]

def test_ping_does_not_need_the_model():
    assert server.app.test_client().get("/api/ping").get_json() == {"app": "vighnax"}


@pytest.mark.skipif(not os.path.isfile(os.path.join(server.ROOT, "models", "world_model.pt")), reason="needs trained model")
def test_older_analysis_stays_explainable_after_a_newer_one():
    """A second run (or a second tab) must not break the panels of the first one."""
    client = server.app.test_client()
    ids = []
    for sample in ("ctu13_s43_neris", "cicids2017_sample"):
        events = [json.loads(line) for line in client.post("/api/stream", data={"sample": sample, "speed": "0"})
                  .get_data(as_text=True).splitlines() if line.strip()]
        ids.append(next(e for e in events if e["type"] == "done")["id"])
    res = client.get(f"/api/explain/{ids[0]}/5")
    assert res.status_code == 200 and res.get_json()["features"]
    assert client.get("/api/explain/unknown-id/5").status_code == 404
