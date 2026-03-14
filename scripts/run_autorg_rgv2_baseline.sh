#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${1:-"${SCRIPT_DIR}/autorg_local.env"}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing env file: ${ENV_FILE}"
  echo "Copy ${SCRIPT_DIR}/autorg_local.env.example to ${SCRIPT_DIR}/autorg_local.env and edit values."
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/features_cache"

echo "[1/5] Preflight check"
python3 "${SCRIPT_DIR}/verify_local_setup.py" \
  --model-folder "${MODEL_FOLDER}" \
  --checkpoint-name "${CHK_NAME}" \
  --seg-pretrained "${SEG_PRETRAINED}" \
  --radgenome-root "${RADGENOME_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --output-root "${OUTPUT_ROOT}"

echo "[2/5] Build test_file.json"
REQ_REPORT=()
if [[ "${STRICT_REPORT:-1}" == "1" ]]; then
  REQ_REPORT+=(--require-report)
fi
python3 "${SCRIPT_DIR}/build_radgenome_test_file.py" \
  --split-json "${SPLIT_JSON}" \
  --case-metadata-json "${CASE_METADATA_JSON}" \
  --radgenome-root "${RADGENOME_ROOT}" \
  --out-json "${TEST_FILE_JSON}" \
  "${REQ_REPORT[@]}"

echo "[3/5] Inference with original test_llm.py"
pushd "${REPO_ROOT}/AutoRG_Brain" >/dev/null
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 test_llm.py \
  --eval_mode given_mask \
  --model_folder "${MODEL_FOLDER}" \
  -chk "${CHK_NAME}" \
  -seg_pretrained "${SEG_PRETRAINED}" \
  -test "${TEST_FILE_JSON}" \
  -o "${OUTPUT_ROOT}" \
  --num_threads_preprocessing "${NUM_THREADS_PREPROCESS:-6}" \
  --num_threads_nifti_save "${NUM_THREADS_NIFTI_SAVE:-2}"
popd >/dev/null

echo "[4/5] Evaluate local metrics (+ optional Table2 hooks)"
python3 "${SCRIPT_DIR}/eval_table2_metrics.py" \
  --pred-json "${PRED_JSON}" \
  --test-json "${TEST_FILE_JSON}" \
  --pred-field "${PRED_FIELD:-pred_global_report}" \
  --out-json "${METRIC_JSON}" \
  --radgraph-cmd "${RADGRAPH_CMD:-}" \
  --ratescore-cmd "${RATESCORE_CMD:-}" \
  --radcliq-cmd "${RADCLIQ_CMD:-}"

echo "[5/5] Done"
echo "Prediction file: ${PRED_JSON}"
echo "Metrics file: ${METRIC_JSON}"
