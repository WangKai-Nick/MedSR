#!/bin/bash

set -euo pipefail

OUTPUT=${1:-}
CHOICE=${2:-}
DATA_PERCENTAGE=${3:-}
SEED=${4:-}

if [ -z "${OUTPUT}" ] || [ -z "${CHOICE}" ] || [ -z "${DATA_PERCENTAGE}" ] || [ -z "${SEED}" ]; then
    echo "Usage: bash scripts/efficiency-all.sh <output_dir> <BUS|ENDO|DERM|BRAIN|CORE|ALL> <data_percentage> <seed>"
    echo "Example: bash scripts/efficiency-all.sh VIP-666-10 CORE 10 666"
    exit 1
fi

BUS=("BUSI" "BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("ISIC" "UWaterlooSkinCancer")
BRAIN=("BTMRI" "BRISC")
CORE=("BUSI" "BTMRI" "ISIC" "Kvasir")

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
    elif [ "${choice}" == "CORE" ]; then
        echo "${CORE[*]}"
    else
        return 1
    fi
}

if [ "${CHOICE}" == "ALL" ]; then
    SELECTED_GROUPS=("BUS" "ENDO" "DERM" "BRAIN")
else
    if ! get_datasets_for_choice "${CHOICE}" >/dev/null 2>&1; then
        echo "Unknown choice: ${CHOICE}"
        exit 1
    fi
    SELECTED_GROUPS=("${CHOICE}")
fi

DATASETS=()
for GROUP in "${SELECTED_GROUPS[@]}"; do
    DATASETS_STR=$(get_datasets_for_choice "${GROUP}")
    read -r -a GROUP_DATASETS <<< "${DATASETS_STR}"
    DATASETS+=("${GROUP_DATASETS[@]}")
done

if [ "${DATA_PERCENTAGE}" != "100" ]; then
    MISSING=()
    for DATASET in "${DATASETS[@]}"; do
        PROMPT_DIR="data/${DATASET}/Prompts_Folder"
        TRAIN_TEXT="${PROMPT_DIR}/Train_text_${DATA_PERCENTAGE}.xlsx"
        VAL_TEXT="${PROMPT_DIR}/Val_text_${DATA_PERCENTAGE}.xlsx"
        if [ ! -f "${TRAIN_TEXT}" ] || [ ! -f "${VAL_TEXT}" ]; then
            MISSING+=("${DATASET}")
        fi
    done

    if [ "${#MISSING[@]}" -gt 0 ]; then
        echo "Missing ${DATA_PERCENTAGE}% prompt split files for:"
        printf '  - %s\n' "${MISSING[@]}"
        echo
        echo "Each listed dataset needs both:"
        echo "  data/<DATASET>/Prompts_Folder/Train_text_${DATA_PERCENTAGE}.xlsx"
        echo "  data/<DATASET>/Prompts_Folder/Val_text_${DATA_PERCENTAGE}.xlsx"
        exit 1
    fi
fi

SUMMARY_DIR="${OUTPUT}/${CHOICE}_${DATA_PERCENTAGE}_efficiency"
SUMMARY_TXT="${SUMMARY_DIR}/summary.txt"
METRICS_TSV="${SUMMARY_DIR}/metrics.tsv"
mkdir -p "${SUMMARY_DIR}"

{
    echo "===================================="
    echo "Efficiency All Summary"
    echo "Choice: ${CHOICE}"
    echo "Data percentage: ${DATA_PERCENTAGE}"
    echo "Seed: ${SEED}"
    echo "Datasets: ${DATASETS[*]}"
    echo "===================================="
} > "${SUMMARY_TXT}"

echo -e "dataset\tdata_percentage\tseed\tdsc\tnsd" > "${METRICS_TSV}"

extract_metrics() {
    local output_dir=$1
    local dataset_name=$2
    local data_percentage=$3
    local seed=$4
    local effective_dataset="${dataset_name}"
    if [ "${data_percentage}" != "100" ]; then
        effective_dataset="${dataset_name}_${data_percentage}"
    fi

    local csv_path="${output_dir}/${effective_dataset}/seg_results/seed${seed}/test_Prompt-original.csv"
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
    print(f"DSC={sum(dsc) / len(dsc) * 100:.2f} NSD={sum(nsd) / len(nsd) * 100:.2f} CSV={csv_path}")
PY
}

for DATASET in "${DATASETS[@]}"; do
    CONFIG=${DATASET}
    echo
    echo "====================================" | tee -a "${SUMMARY_TXT}"
    echo "Dataset: ${DATASET} | data_percentage=${DATA_PERCENTAGE} | seed=${SEED}" | tee -a "${SUMMARY_TXT}"
    echo "====================================" | tee -a "${SUMMARY_TXT}"

    python train.py \
        --config-file "configs/${CONFIG}.yaml" \
        --output-dir "${OUTPUT}" \
        --data_percentage "${DATA_PERCENTAGE}" \
        --seed "${SEED}"

    python test.py \
        --config-file "configs/${CONFIG}.yaml" \
        --output-dir "${OUTPUT}" \
        --source_dataset "${DATASET}" \
        --data_percentage "${DATA_PERCENTAGE}" \
        --seed "${SEED}"

    python utils/eval.py \
        --config-file "configs/${CONFIG}.yaml" \
        --output-dir "${OUTPUT}" \
        --data_percentage "${DATA_PERCENTAGE}" \
        --seed "${SEED}"

    METRICS=$(extract_metrics "${OUTPUT}" "${DATASET}" "${DATA_PERCENTAGE}" "${SEED}")
    echo "[dataset=${DATASET} data_percentage=${DATA_PERCENTAGE} seed=${SEED}] ${METRICS}" | tee -a "${SUMMARY_TXT}"

    DSC_VAL=$(echo "${METRICS}" | sed -n 's/.*DSC=\([^ ]*\).*/\1/p')
    NSD_VAL=$(echo "${METRICS}" | sed -n 's/.*NSD=\([^ ]*\).*/\1/p')
    echo -e "${DATASET}\t${DATA_PERCENTAGE}\t${SEED}\t${DSC_VAL}\t${NSD_VAL}" >> "${METRICS_TSV}"
done

echo
echo "Saved summary to: ${SUMMARY_TXT}"
echo "Saved metrics to: ${METRICS_TSV}"
