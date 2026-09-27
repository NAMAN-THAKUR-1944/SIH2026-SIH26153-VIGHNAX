#!/usr/bin/env bash
# Download every dataset named in the PS into datasets/ (about 9 GB in total).
#
#   bash scripts/fetch_datasets.sh                 # public datasets only
#   CIC_FIRST=... CIC_LAST=... CIC_EMAIL=... CIC_ORG=... CIC_TITLE=Student CIC_COUNTRY=India \
#   LANL_URL=https://csr.lanl.gov/data-fence/<token>/cyber1 \
#   bash scripts/fetch_datasets.sh                 # + registration-gated datasets
#
# Gated datasets (no personal data is stored in this repository):
#   * CIC-IDS2017, CICIoT2023 - the UNB CIC download form (first/last name, email, organisation,
#     job title, country) at https://www.unb.ca/cic/datasets/ . This script submits the same form.
#   * LANL cyber1 - fill the form at https://csr.lanl.gov/data/cyber1/ and pass the base of the
#     signed download links it shows as LANL_URL.
set -u
cd "$(dirname "$0")/.."
mkdir -p datasets && cd datasets

# ---------------------------------------------------------------- CTU-13 (public)
bash ../scripts/fetch_ctu13.sh

# ---------------------------------------------------------------- CSE-CIC-IDS2018 (public AWS S3)
mkdir -p cicids2018
S3="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
for f in Wednesday-14-02-2018 Thursday-15-02-2018 Thursday-22-02-2018 Wednesday-28-02-2018 Friday-02-03-2018; do
  curl -s --retry 5 -C - -o "cicids2018/$f.csv" "$S3/${f}_TrafficForML_CICFlowMeter.csv" && echo "done cicids2018/$f" &
done
wait

# ---------------------------------------------------------------- UNSW-NB15 (public SharePoint folder)
mkdir -p unswnb15
SHARE="https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
curl -s -L -c unswnb15/jar.txt -b unswnb15/jar.txt -A "Mozilla/5.0" -o /dev/null "$SHARE"
unsw() {
  local src="/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset/CSV Files/$1"
  curl -s --retry 5 -b unswnb15/jar.txt -A "Mozilla/5.0" -G --data-urlencode "@a='$src'" -o "unswnb15/$2" \
    "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_api/web/GetFileByServerRelativePath(decodedurl=@a)/\$value" \
    && echo "done unswnb15/$2"
}
unsw NUSW-NB15_features.csv features.csv
for i in 1 2 3 4; do unsw "UNSW-NB15_$i.csv" "UNSW-NB15_$i.csv" & done
wait

# ---------------------------------------------------------------- DARPA 1999 (public MIT LL archive)
mkdir -p darpa1999
LL="https://archive.ll.mit.edu/ideval"
curl -s -o darpa1999/master_identifications.list "$LL/files/master_identifications.list"
for w in week4 week5; do for d in monday tuesday wednesday thursday friday; do
  [ "$w/$d" = "week4/tuesday" ] && continue        # never published
  curl -s --retry 5 -C - -o "darpa1999/${w}_${d}.tcpdump.gz" "$LL/data/1999/testing/$w/$d/inside.tcpdump.gz" && echo "done darpa1999/${w}_${d}" &
done; done
wait

# ---------------------------------------------------------------- CIC-IDS2017 + CICIoT2023 (UNB form)
if [ -n "${CIC_EMAIL:-}" ]; then
  cic_register() {  # $1 = base URL, $2 = cookie jar
    curl -s -c "$2" -b "$2" -A "Mozilla/5.0" -e "$1/" -F "first_name=$CIC_FIRST" -F "last_name=$CIC_LAST" \
      -F "email=$CIC_EMAIL" -F "institution=$CIC_ORG" -F "job_title=${CIC_TITLE:-Student}" \
      -F "country=${CIC_COUNTRY:-India}" "$1/insert.php"; echo
  }
  mkdir -p cicids2017
  B17="https://cicresearch.ca/CICDataset/CIC-IDS-2017"
  cic_register "$B17" cicids2017/cookies.txt
  curl -s -b cicids2017/cookies.txt -A "Mozilla/5.0" -o cicids2017/GeneratedLabelledFlows.zip \
    "$B17/download.php?file=CIC-IDS-2017%2FCSVs%2FGeneratedLabelledFlows.zip" && echo "done cicids2017"

  mkdir -p ciciot2023
  BIOT="https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023"
  cic_register "$BIOT" ciciot2023/cookies.txt
  # The server streams whole captures (no Range support) and times out on parallel requests:
  # fetch one at a time and keep the first 200 MB of each.
  for f in Benign_Final/BenignTraffic Recon-PortScan/Recon-PortScan Recon-HostDiscovery/Recon-HostDiscovery \
           DictionaryBruteForce/DictionaryBruteForce Backdoor_Malware/Backdoor_Malware Uploading_Attack/Uploading_Attack \
           CommandInjection/CommandInjection Mirai-greeth_flood/Mirai-greeth_flood DDoS-ICMP_Flood/DDoS-ICMP_Flood; do
    n=$(basename "$f")
    curl -s -b ciciot2023/cookies.txt -A "Mozilla/5.0" "$BIOT/download.php?file=PCAP%2F${f//\//%2F}.pcap" \
      | head -c 209715200 > "ciciot2023/$n.pcap"; echo "done ciciot2023/$n"
  done
else
  echo "skipping CIC-IDS2017 / CICIoT2023: set CIC_FIRST, CIC_LAST, CIC_EMAIL, CIC_ORG to register"
fi

# ---------------------------------------------------------------- LANL cyber1 (form -> signed URL)
if [ -n "${LANL_URL:-}" ]; then
  mkdir -p lanl
  curl -s -o lanl/redteam.txt.gz "$LANL_URL/redteam.txt.gz" && echo "done lanl/redteam"
  curl -s --retry 8 -C - -o lanl/flows.txt.gz "$LANL_URL/flows.txt.gz" && gzip -t lanl/flows.txt.gz && echo "done lanl/flows"
else
  echo "skipping LANL: fill the form at https://csr.lanl.gov/data/cyber1/ and set LANL_URL"
fi
