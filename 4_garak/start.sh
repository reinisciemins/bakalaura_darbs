set -u

[[ -d ".venv" ]] || { echo "ERROR: .venv not found. Run from ~/Documents/practical/4_garak"; exit 1; }
[[ -f ".env.local" ]] || { echo "ERROR: .env.local not found. Required: OPENROUTER_API_KEY; optional: HF_TOKEN"; exit 1; }

source .venv/bin/activate
set -a; source .env.local; set +a

: "${OPENROUTER_API_KEY:?ERROR: OPENROUTER_API_KEY is missing in .env.local}"

[[ -n "${HF_TOKEN:-}" ]] && {
  export HF_INFERENCE_TOKEN="${HF_INFERENCE_TOKEN:-$HF_TOKEN}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
}

export OPENROUTER_API_BASE="${OPENROUTER_API_BASE:-https://openrouter.ai/api/v1}"
export OR_SITE_URL="${OR_SITE_URL:-http://localhost}"
export OR_APP_NAME="${OR_APP_NAME:-bachelor-thesis-garak}"

mkdir -p results/logs results/reports

run_ts="$(date +%Y%m%d_%H%M%S)"
status_file="results/run_status_${run_ts}.csv"
garak_dir="$HOME/.local/share/garak/garak_runs"

echo "model,probe,exit_code,jsonl_report,html_report" > "$status_file"

models=(
  "gpt|litellm|openrouter/openai/gpt-5.4-nano"
  "gemini|litellm|openrouter/google/gemini-3.1-flash-lite"
  "claude|litellm|openrouter/anthropic/claude-haiku-4.5"
  "deepseek|litellm|openrouter/deepseek/deepseek-v4-flash"
)

probes=(
  "misleading.FalseAssertion"
  "dan.Ablation_Dan_11_0"
)

latest_report() {
  find "$garak_dir" -name "*.$1" -newermt "@$2" 2>/dev/null | sort | tail -n 1
}

for model_data in "${models[@]}"; do
  IFS="|" read -r model_label target_type target_name <<< "$model_data"

  for probe in "${probes[@]}"; do
    safe_probe="${probe//./_}"
    log_file="results/logs/${run_ts}_${model_label}_${safe_probe}.txt"
    started_at="$(date +%s)"

    echo "============================================"
    echo "Running: $model_label | $target_type | $target_name | $probe"
    echo "Log: $log_file"
    echo "============================================"

    set +e
    python -m garak \
      --target_type "$target_type" \
      --target_name "$target_name" \
      --probes "$probe" \
      --generations 1 \
      --parallel_requests 1 \
      --parallel_attempts 1 \
      2>&1 | tee "$log_file"
    exit_code="${PIPESTATUS[0]}"
    set -e

    jsonl_src="$(latest_report jsonl "$started_at")"
    html_src="$(latest_report html "$started_at")"

    jsonl_dst=""
    html_dst=""

    [[ -f "$jsonl_src" ]] && {
      jsonl_dst="results/reports/${run_ts}_${model_label}_${safe_probe}.report.jsonl"
      cp "$jsonl_src" "$jsonl_dst"
    }

    [[ -f "$html_src" ]] && {
      html_dst="results/reports/${run_ts}_${model_label}_${safe_probe}.report.html"
      cp "$html_src" "$html_dst"
    }

    echo "$model_label,$probe,$exit_code,$jsonl_dst,$html_dst" >> "$status_file"
    echo ""
  done
done

echo "============================================"
echo "Garak runs finished."
echo "Status file: $status_file"
echo "Logs: results/logs/"
echo "Reports: results/reports/"
echo "Global Garak log: $HOME/.local/share/garak/garak.log"
echo "============================================"