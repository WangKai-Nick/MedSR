#!/bin/bash

set -euo pipefail

OUTPUT=${1:-}
CHOICE=${2:-}
FIXED_SEED=${3:-}
NUM_RUNS=${4:-1}

if [ -z "${OUTPUT}" ] || [ -z "${CHOICE}" ] || [ -z "${FIXED_SEED}" ]; then
    echo "Usage: bash scripts/domain_generalization-chong.sh <output_dir> <BUS|ENDO|DERM|BRAIN|ALL> <fixed_seed> [num_runs]"
    exit 1
fi

# Predefined dataset lists
BUS=("BUSI" "BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("ISIC" "UWaterlooSkinCancer")
BRAIN=("BTMRI" "BRISC")

get_datasets_for_choice() {
    local choice=$1
    if [ "${choice}" == "BUS" ]; then
        echo "${BUS[*]}"
    elif [ "${choice}" == "ENDO" ]; then
        echo "${ENDO[*]}"
    elif [ "${choice}" == "DERM" ]; then
        echo "${DERM[*]}"
    elif [ "${choice}" == "BRAIN" ]; then
        echo "${BRAIN[*]}"
    else
        return 1
    fi
}

if [ "${CHOICE}" == "ALL" ]; then
    DG_GROUPS=("BUS" "ENDO" "DERM" "BRAIN")
else
    if ! get_datasets_for_choice "${CHOICE}" >/dev/null 2>&1; then
        echo "Unknown choice: ${CHOICE}"
        exit 1
    fi
    DG_GROUPS=("${CHOICE}")
fi

SUMMARY_DIR="${OUTPUT}/${CHOICE}_multi_run"
SUMMARY_TXT="${SUMMARY_DIR}/summary.txt"
METRICS_TSV="${SUMMARY_DIR}/metrics.tsv"

mkdir -p "${SUMMARY_DIR}"

{
    echo "===================================="
    echo "Domain Generalization Multi-run Summary"
    echo "Choice: ${CHOICE}"
    echo "Fixed seed: ${FIXED_SEED}"
    echo "Num runs: ${NUM_RUNS}"
    echo "Groups: ${DG_GROUPS[*]}"
    echo "===================================="
} > "${SUMMARY_TXT}"

echo -e "group\trun\tseed\tsource\ttarget\tdsc\tnsd" > "${METRICS_TSV}"

extract_metrics() {
    local run_output=$1
    local dataset_name=$2
    local seed=$3
    local csv_path
    csv_path="${run_output}/${dataset_name}/seg_results/seed${seed}/test_Prompt-original.csv"

    if [ ! -f "${csv_path}" ]; then
        echo "DSC=NA NSD=NA CSV=${csv_path}"
        return
    fi

    python - "${csv_path}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
dsc = []
nsd = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dsc.append(float(row["DSC"]))
        nsd.append(float(row["NSD"]))

if not dsc:
    print(f"DSC=NA NSD=NA CSV={csv_path}")
else:
    print(f"DSC={sum(dsc)/len(dsc)*100:.2f} NSD={sum(nsd)/len(nsd)*100:.2f} CSV={csv_path}")
PY
}

append_metrics_record() {
    local group_name=$1
    local run_id=$2
    local seed=$3
    local source_name=$4
    local target_name=$5
    local metrics_line=$6

    local dsc_val
    local nsd_val
    dsc_val=$(echo "${metrics_line}" | sed -n 's/.*DSC=\([^ ]*\).*/\1/p')
    nsd_val=$(echo "${metrics_line}" | sed -n 's/.*NSD=\([^ ]*\).*/\1/p')

    echo -e "${group_name}\t${run_id}\t${seed}\t${source_name}\t${target_name}\t${dsc_val}\t${nsd_val}" >> "${METRICS_TSV}"
}

for GROUP in "${DG_GROUPS[@]}"; do
    DATASETS_STR=$(get_datasets_for_choice "${GROUP}")
    read -r -a DATASETS <<< "${DATASETS_STR}"
    SOURCE=${DATASETS[0]}

    {
        echo
        echo "===================================="
        echo "GROUP: ${GROUP}"
        echo "SOURCE: ${SOURCE}"
        echo "TARGETS: ${DATASETS[*]}"
        echo "===================================="
    } | tee -a "${SUMMARY_TXT}"

    for ((run_idx=0; run_idx<NUM_RUNS; run_idx++)); do
        CURRENT_SEED=${FIXED_SEED}
        RUN_ID=$((run_idx + 1))
        RUN_OUTPUT="${SUMMARY_DIR}/${GROUP}/run_${RUN_ID}"

        {
            echo
            echo "####################################"
            echo "Group ${GROUP} | Run ${RUN_ID} / ${NUM_RUNS}"
            echo "Seed: ${CURRENT_SEED}"
            echo "Run output: ${RUN_OUTPUT}"
            echo "####################################"
            echo "SOURCE DATASET (fully supervised): ${SOURCE}"
            echo "TARGET DATASETS: ${DATASETS[*]:1}"
            echo
        } | tee -a "${SUMMARY_TXT}"

        CONFIG=${SOURCE}

        python train.py \
            --config-file "configs/${CONFIG}.yaml" \
            --output-dir "${RUN_OUTPUT}" \
            --seed "${CURRENT_SEED}"

        echo "Evaluating SOURCE=${SOURCE} -> TARGET=${SOURCE} (in-domain)" | tee -a "${SUMMARY_TXT}"

        python test.py \
            --config-file "configs/${SOURCE}.yaml" \
            --output-dir "${RUN_OUTPUT}" \
            --source_dataset "${SOURCE}" \
            --seed "${CURRENT_SEED}"

        python utils/eval.py \
            --config-file "configs/${SOURCE}.yaml" \
            --output-dir "${RUN_OUTPUT}" \
            --seed "${CURRENT_SEED}"

        SOURCE_METRICS=$(extract_metrics "${RUN_OUTPUT}" "${SOURCE}" "${CURRENT_SEED}")
        echo "[group=${GROUP} run=${RUN_ID} seed=${CURRENT_SEED}] ${SOURCE} -> ${SOURCE} ${SOURCE_METRICS}" | tee -a "${SUMMARY_TXT}"
        append_metrics_record "${GROUP}" "${RUN_ID}" "${CURRENT_SEED}" "${SOURCE}" "${SOURCE}" "${SOURCE_METRICS}"

        for TARGET in "${DATASETS[@]:1}"; do
            echo "Evaluating SOURCE=${SOURCE} -> TARGET=${TARGET}" | tee -a "${SUMMARY_TXT}"

            python test.py \
                --config-file "configs/${TARGET}.yaml" \
                --output-dir "${RUN_OUTPUT}" \
                --source_dataset "${SOURCE}" \
                --seed "${CURRENT_SEED}"

            python utils/eval.py \
                --config-file "configs/${TARGET}.yaml" \
                --output-dir "${RUN_OUTPUT}" \
                --seed "${CURRENT_SEED}"

            TARGET_METRICS=$(extract_metrics "${RUN_OUTPUT}" "${TARGET}" "${CURRENT_SEED}")
            echo "[group=${GROUP} run=${RUN_ID} seed=${CURRENT_SEED}] ${SOURCE} -> ${TARGET} ${TARGET_METRICS}" | tee -a "${SUMMARY_TXT}"
            append_metrics_record "${GROUP}" "${RUN_ID}" "${CURRENT_SEED}" "${SOURCE}" "${TARGET}" "${TARGET_METRICS}"
        done
    done
done

python - "${METRICS_TSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import math
import sys
from collections import defaultdict

metrics_tsv = sys.argv[1]
summary_txt = sys.argv[2]

groups = defaultdict(lambda: {"source": None, "targets": defaultdict(lambda: {"dsc": [], "nsd": []})})

with open(metrics_tsv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["dsc"] == "NA" or row["nsd"] == "NA":
            continue
        group = row["group"]
        source = row["source"]
        target = row["target"]
        groups[group]["source"] = source
        groups[group]["targets"][target]["dsc"].append(float(row["dsc"]))
        groups[group]["targets"][target]["nsd"].append(float(row["nsd"]))

def mean_std(values):
    if not values:
        return ("NA", "NA")
    mean_val = sum(values) / len(values)
    if len(values) == 1:
        std_val = 0.0
    else:
        var = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(var)
    return (f"{mean_val:.2f}", f"{std_val:.2f}")

with open(summary_txt, "a", encoding="utf-8") as f:
    f.write("\n====================================\n")
    f.write("Aggregated Mean ± Std\n")
    f.write("====================================\n")
    for group in sorted(groups.keys()):
        source = groups[group]["source"]
        f.write(f"\n[GROUP={group}] SOURCE={source}\n")
        for target in sorted(groups[group]["targets"].keys()):
            dsc_mean, dsc_std = mean_std(groups[group]["targets"][target]["dsc"])
            nsd_mean, nsd_std = mean_std(groups[group]["targets"][target]["nsd"])
            f.write(
                f"{source} -> {target} | "
                f"DSC={dsc_mean}±{dsc_std} | "
                f"NSD={nsd_mean}±{nsd_std}\n"
            )
PY

echo
echo "Saved summary to: ${SUMMARY_TXT}"
