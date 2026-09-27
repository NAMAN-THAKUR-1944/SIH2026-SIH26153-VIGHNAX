#!/usr/bin/env bash
# Fetch CTU-13 labelled bidirectional flows + header-truncated full PCAPs.
# Source: Stratosphere Lab, CTU Prague (CC-BY). https://www.stratosphereips.org/datasets-ctu13
cd "$(dirname "$0")/.." && mkdir -p datasets/ctu13 && cd datasets/ctu13
B="https://mcfp.felk.cvut.cz/publicDatasets"
get() { curl -s -L --retry 8 --retry-all-errors -C - -o "$2" "$B/$1" && echo "done $2 $(du -h "$2" | cut -f1)"; }
export -f get; export B
cat <<LIST | xargs -P 5 -L 1 bash -c 'get "$0" "$1"'
CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow s52.binetflow
CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow s46.binetflow
CTU-Malware-Capture-Botnet-48/detailed-bidirectional-flow-labels/capture20110816-2.binetflow s48.binetflow
CTU-Malware-Capture-Botnet-53/detailed-bidirectional-flow-labels/capture20110819.binetflow s53.binetflow
CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow s47.binetflow
CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow s51.binetflow
CTU-Malware-Capture-Botnet-43/detailed-bidirectional-flow-labels/capture20110811.binetflow s43.binetflow
CTU-Malware-Capture-Botnet-52/capture20110818-2.truncated.pcap.bz2 s52.pcap.bz2
CTU-Malware-Capture-Botnet-46/capture20110815-2.truncated.pcap.bz2 s46.pcap.bz2
CTU-Malware-Capture-Botnet-48/capture20110816-2.truncated.pcap.bz2 s48.pcap.bz2
CTU-Malware-Capture-Botnet-53/capture20110819.truncated.pcap.bz2 s53.pcap.bz2
CTU-Malware-Capture-Botnet-47/capture20110816.truncated.pcap.bz2 s47.pcap.bz2
CTU-Malware-Capture-Botnet-51/capture20110818.truncated.pcap.bz2 s51.pcap.bz2
CTU-Malware-Capture-Botnet-43/capture20110811.truncated.pcap.bz2 s43.pcap.bz2
CTU-Malware-Capture-Botnet-54/detailed-bidirectional-flow-labels/capture20110815-3.binetflow s54.binetflow
CTU-Malware-Capture-Botnet-54/capture20110815-3.truncated.pcap.bz2 s54.pcap.bz2
LIST
echo ALL_DONE
