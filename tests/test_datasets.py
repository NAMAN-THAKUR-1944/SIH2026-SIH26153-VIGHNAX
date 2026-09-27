"""Format detection and dataset-specific quirks of the PS datasets."""

import pandas as pd

from src.data.datasets import cic2017_frame, detect_flow_format, load_flows_any

CIC17_HEADER = ("Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration, "
                "Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Label")


def test_cic2017_afternoon_clock_and_minute_spread():
    raw = pd.DataFrame({
        "Flow ID": ["a", "b", "c", "d"], "Source IP": ["10.0.0.1"] * 4, "Source Port": [1, 2, 3, 4],
        "Destination IP": ["10.0.0.2"] * 4, "Destination Port": [80] * 4, "Protocol": [6] * 4,
        "Timestamp": ["7/7/2017 9:00", "7/7/2017 1:30", "7/7/2017 1:30", "7/7/2017 1:30"],
        "Flow Duration": [1000] * 4, "Total Fwd Packets": [1] * 4, "Total Backward Packets": [1] * 4,
        "Total Length of Fwd Packets": [10] * 4, "Total Length of Bwd Packets": [10] * 4,
        "Label": ["BENIGN", "PortScan", "PortScan", "PortScan"],
    })
    f = cic2017_frame(raw).sort_values("start_time")
    hours = pd.to_datetime(f["start_time"], unit="s").dt.hour.tolist()
    assert hours == [9, 13, 13, 13]                      # "1:30" is 13:30 (12-hour clock, no AM/PM)
    pm = f["start_time"].to_numpy()[1:]
    assert len(set(pm)) == 3 and pm.max() - pm.min() < 60  # spread inside the minute, in file order


def test_detect_formats(tmp_path):
    cases = {
        "ctu": "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
               "2011/08/18 15:39:35.08,1.0,tcp,1.1.1.1,1,->,2.2.2.2,80,S_,0,0,1,60,60,flow=Background\n",
        "cic2017": CIC17_HEADER + "\nx,1.1.1.1,1,2.2.2.2,80,6,7/7/2017 9:00,10,1,1,10,10,BENIGN\n",
        "cic": "Dst Port,Protocol,Timestamp,Flow Duration,Label\n443,6,02/03/2018 08:47:38,10,Benign\n",
        "unsw": "59.166.0.1,18247,149.171.126.4,7662,tcp,FIN," + ",".join(["0"] * 41) + ",,0\n",  # 49 fields
        "lanl": "1,9,C3090,N10471,C3420,N46,6,3,144\n",
    }
    for fmt, text in cases.items():
        p = tmp_path / f"{fmt}.csv"
        p.write_text(text)
        assert detect_flow_format(str(p)) == fmt, fmt


def test_lanl_rows_load(tmp_path):
    p = tmp_path / "lanl.csv"
    p.write_text("time,duration,src,sport,dst,dport,proto,packets,bytes,label\n"
                 "10,1,C1,N1,C2,445,6,3,144,lanl-redteam-lateral-movement\n20,2,C3,N2,C4,80,6,5,500,normal\n")
    f = load_flows_any(str(p))
    assert len(f) == 2 and f["dst_port"].tolist() == [445, 80] and f["src_ip"].tolist() == ["C1", "C3"]
