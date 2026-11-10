#!/usr/bin/env bash
#
# Example: generate VisCodeBench predictions with a coding agent
# (OpenAI Codex CLI or Claude Code), one sample at a time.
#
# This is a reference recipe to read and adapt, not a turnkey tool. It runs a
# coding agent that writes files and executes code inside each sample folder,
# so read it before running it, and run it in an isolated environment. The
# Docker image in this repo is set up for it. See docs/tutorial.md.
#
# Usage:
#   AGENT=codex  bash run_agent.sh
#   AGENT=claude MODEL=<model> MAX_SAMPLES=5 bash run_agent.sh
#
# Settings (environment variables):
#   AGENT              codex or claude                                (required)
#   MODEL              model name passed to the agent                 (default: the agent's default)
#   DATA_DIR           folder containing the subset folders           (default: data/codabench)
#   SUBSETS            space-separated subsets to process             (default: all)
#   MAX_SAMPLES        process at most this many samples, then stop.  (default: 0 = all)
#                      Already-done samples don't count, so rerunning
#                      with MAX_SAMPLES=5 processes the next 5.
#   PYTHON_BIN         interpreter the agent should run scripts with  (default: python on PATH)
#   RUN_DIR            this run's copies of the samples, with the     (default: runs/<AGENT>/<MODEL>,
#                      same layout as DATA_DIR (see below)              or runs/<AGENT>/default without MODEL)
#   SLEEP_SECONDS      pause between samples                          (default: 5)
#   ON_USAGE_LIMIT     what to do when the agent reports a usage or   (default: wait)
#                      rate limit: "wait" (sleep, then retry the same
#                      sample) or "stop" (exit; rerun later to resume)
#   USAGE_LIMIT_WAIT   seconds to sleep before retrying after a limit (default: 1800)
#   AGENT_FULL_ACCESS  1: no agent sandbox or permission checks       (default: 0; the Docker image sets 1)
#                        (Codex --sandbox danger-full-access, Claude --dangerously-skip-permissions).
#                        Only inside a container/VM that is itself the isolation boundary.
#                      0: Codex --sandbox workspace-write (Linux needs
#                        unprivileged user namespaces); Claude may edit files in the
#                        sample folder and run only PYTHON_BIN.
#
# DATA_DIR is never modified. Before the agent works on a sample, the sample
# folder is copied to RUN_DIR/<subset>/<sample_id>/ and the agent works only on
# that copy, so different agents and models never see each other's answers:
#
#   RUN_DIR/<subset>/<sample_id>/
#   ├── <sample_id>.py        the agent's answer
#   ├── outputs/              the agent's images
#   ├── ...                   the sample's data files
#   └── .run/                 <AGENT>.log, elapsed_seconds.txt, done
#
# RUN_DIR then has the same layout as DATA_DIR, so it can be logged directly:
#   python -m t2v_eval.batch_log --data_dir RUN_DIR --zip
#
# The script can be stopped (Ctrl+C) and rerun: samples that finished are
# skipped. Delete RUN_DIR/<subset>/<sample_id>/ to redo one sample from scratch.

set -uo pipefail

AGENT="${AGENT:-}"
MODEL="${MODEL:-}"
DATA_DIR="${DATA_DIR:-data/codabench}"
SUBSETS="${SUBSETS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
# Folder-safe model name, e.g. "openai/gpt-x:latest" -> "openai_gpt-x_latest".
MODEL_TAG=$(printf '%s' "${MODEL:-default}" | tr -c 'A-Za-z0-9._-' '_')
RUN_DIR="${RUN_DIR:-runs/$AGENT/$MODEL_TAG}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
ON_USAGE_LIMIT="${ON_USAGE_LIMIT:-wait}"
USAGE_LIMIT_WAIT="${USAGE_LIMIT_WAIT:-1800}"
AGENT_FULL_ACCESS="${AGENT_FULL_ACCESS:-0}"

die() { echo "$*" >&2; exit 1; }
case "$AGENT" in
    codex|claude) command -v "$AGENT" >/dev/null || die "'$AGENT' is not installed or not on PATH." ;;
    *) die "Set AGENT to codex or claude, e.g. AGENT=codex bash run_agent.sh" ;;
esac
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]]      || die "MAX_SAMPLES must be a non-negative integer, got '$MAX_SAMPLES'"
[[ "$USAGE_LIMIT_WAIT" =~ ^[0-9]+$ ]] || die "USAGE_LIMIT_WAIT must be a number of seconds, got '$USAGE_LIMIT_WAIT'"
[[ "$ON_USAGE_LIMIT" =~ ^(wait|stop)$ ]] || die "ON_USAGE_LIMIT must be wait or stop, got '$ON_USAGE_LIMIT'"
[[ "$AGENT_FULL_ACCESS" =~ ^[01]$ ]]  || die "AGENT_FULL_ACCESS must be 0 or 1, got '$AGENT_FULL_ACCESS'"

if [ "$AGENT_FULL_ACCESS" = 1 ]; then
    echo "WARNING: AGENT_FULL_ACCESS=1 turns off the agent's sandbox and permission checks."
    echo "         The agent can read, write and run anything this user can."
    echo "         Only continue if you are inside a disposable container or VM."
fi

trap 'echo; echo "Interrupted. Rerun to continue where you left off."; exit 130' INT

# --- Helpers ----------------------------------------------------------------

# Count non-blank, non-comment lines between the visualisation markers.
filled_lines() {
    awk '/##START VISUALISATION CODE/{f=1; next} /##END VISUALISATION CODE/{f=0} f' "$1" \
        | grep -cvE '^[[:space:]]*(#.*)?$'
}

# Messages that mean "not logged in / bad credentials": every sample would fail.
# Claude Code: "Not logged in · Please run /login", "Invalid API key". Codex: "401 Unauthorized".
AUTH_PATTERN='Not logged in|Please run /login|Invalid API key|401 Unauthorized|authentication_failed|token has expired'
# Messages that mean "out of credit": waiting won't help.
BILLING_PATTERN='Credit balance is too low|spend limit|billing|insufficient_quota|exceeded your current quota'
# Messages that mean "usage or rate limit": waiting will help.
# Claude Code: "You've hit your session/weekly/... limit · resets ...", "API Error: ... (429)",
#   "temporarily limiting requests", "529 Overloaded". Codex: "You've hit your usage limit", 429s.
LIMIT_PATTERN="hit your .*limit|usage limit|rate limit|Rate limit|\(429\)|429 Too Many|Too Many Requests|temporarily limiting requests|Overloaded"

run_agent() {  # run_agent <sample_dir> <prompt> <log_file>
    local sample_dir="$1" prompt="$2" log="$3"
    local model_args=()
    [ -n "$MODEL" ] && model_args=(--model "$MODEL")

    if [ "$AGENT" = codex ]; then
        local sandbox=workspace-write
        [ "$AGENT_FULL_ACCESS" = 1 ] && sandbox=danger-full-access
        codex exec ${model_args[@]+"${model_args[@]}"} \
            --cd "$sample_dir" \
            --skip-git-repo-check \
            --sandbox "$sandbox" \
            "$prompt" > "$log" 2>&1
    else
        local perm_args=(--permission-mode acceptEdits --allowedTools "Bash(${PYTHON_BIN} *)")
        [ "$AGENT_FULL_ACCESS" = 1 ] && perm_args=(--dangerously-skip-permissions)
        (cd "$sample_dir" && claude -p "$prompt" ${model_args[@]+"${model_args[@]}"} "${perm_args[@]}") > "$log" 2>&1
    fi
}

build_prompt() {
    local sample_id="$1" vis_request="$2"
    cat <<EOF
Given data files and a code context file '${sample_id}.py'.

Visualisation request:
'${vis_request}'

Requirements:
(1) Generate a visualisation (chart or map) based on the data files, code context, and the request.
(2) The generated visualisation must be relevant.
(3) Use Python libraries such as matplotlib, seaborn, or geopandas for visualisation.
(4) The visualisation should be clear, well-labelled, and visually appealing, with no overlapping text or overflowing elements.
(5) For maps, use the GeoJSON file "countries.geo.json" for country boundaries.
(6) For maps, display data-present areas in suitable colours, and data-absent areas in white with a hatched style and grey border where applicable.
(7) Save the visualisation as PNG image file(s) at './outputs/output_{i}.png' unless the request specifies otherwise.
(8) Use a non-interactive backend and do not display the visualisation, to avoid timeouts.
(9) Put the code between the '##START VISUALISATION CODE' and '##END VISUALISATION CODE' markers, which are already at the end of the code context file.
(10) Use relative paths to access data files and the code context file, and to save outputs.
(11) Execute the visualisation program using the Python interpreter at: '${PYTHON_BIN}'
EOF
}

# --- Main loop --------------------------------------------------------------

processed=0
for request_path in "$DATA_DIR"/*/*/*.vis-request.txt; do
    [ -e "$request_path" ] || die "No samples found under $DATA_DIR"

    sample_dir=$(dirname "$request_path")
    sample_id=$(basename "$request_path" .vis-request.txt)
    subset=$(basename "$(dirname "$sample_dir")")

    if [ -n "$SUBSETS" ] && [[ " $SUBSETS " != *" $subset "* ]]; then
        continue
    fi

    work_dir="$RUN_DIR/$subset/$sample_id"   # this run's copy of the sample
    meta_dir="$work_dir/.run"                # log, timing and done marker
    if [ -f "$meta_dir/done" ]; then
        echo "Skipping $subset/$sample_id (already done)"
        continue
    fi

    # The original must be untouched, or the agent would start from someone
    # else's answer.
    if [ "$(filled_lines "$sample_dir/$sample_id.py")" -gt 0 ]; then
        echo "Skipping $subset/$sample_id: the original in $sample_dir already has code"
        echo "  between the markers. Restore it from t2v-public-data.zip."
        continue
    fi

    if [ "$MAX_SAMPLES" -gt 0 ] && [ "$processed" -ge "$MAX_SAMPLES" ]; then
        echo "Reached MAX_SAMPLES=$MAX_SAMPLES, stopping."
        break
    fi
    processed=$((processed + 1))
    prompt=$(build_prompt "$sample_id" "$(cat "$request_path")")

    # Retry the same sample while the agent reports a usage/rate limit.
    while true; do
        echo "Processing $subset/$sample_id"
        # Start every attempt from a fresh copy of the original sample.
        rm -rf "$work_dir"
        mkdir -p "$(dirname "$work_dir")"
        cp -R "$sample_dir" "$work_dir"
        mkdir -p "$meta_dir"
        log="$meta_dir/$AGENT.log"

        start=$(date +%s)
        run_agent "$work_dir" "$prompt" "$log"
        status=$?
        echo $(( $(date +%s) - start )) > "$meta_dir/elapsed_seconds.txt"

        # Agents can exit 0 without doing anything, so also check that the
        # visualisation region was actually filled in.
        filled=$(filled_lines "$work_dir/$sample_id.py")
        if [ "$status" -eq 0 ] && [ "$filled" -gt 0 ]; then
            touch "$meta_dir/done"
            echo "  done ($(cat "$meta_dir/elapsed_seconds.txt")s)"
            break
        fi

        if grep -qE "$AUTH_PATTERN" "$log"; then
            echo "  $AGENT is not logged in or its credentials are invalid, see $log."
            echo "  Log in first (docs/tutorial.md, step 4). Stopping."
            exit 1
        fi
        if grep -qiE "$BILLING_PATTERN" "$log"; then
            echo "  $AGENT reports a billing or credit problem, see $log. Stopping."
            exit 1
        fi
        if grep -qE "$LIMIT_PATTERN" "$log"; then
            echo "  $AGENT hit a usage or rate limit:"
            echo "    $(grep -m1 -E "$LIMIT_PATTERN" "$log" | cut -c1-160)"
            if [ "$ON_USAGE_LIMIT" = stop ]; then
                echo "Stopping (ON_USAGE_LIMIT=stop). Rerun later to continue from this sample."
                exit 2
            fi
            echo "  waiting ${USAGE_LIMIT_WAIT}s until $(date -d "@$(( $(date +%s) + USAGE_LIMIT_WAIT ))" '+%H:%M' 2>/dev/null || date -r "$(( $(date +%s) + USAGE_LIMIT_WAIT ))" '+%H:%M'), then retrying this sample (Ctrl+C to stop)"
            sleep "$USAGE_LIMIT_WAIT"
            continue
        fi
        if [ "$AGENT" = codex ] && grep -q "bwrap:" "$log"; then
            echo
            echo "Codex's sandbox (bubblewrap) cannot run on this machine, see $log."
            echo "Ask your admin to allow unprivileged user namespaces, or run this script"
            echo "inside the Docker container (AGENT_FULL_ACCESS=1). Stopping."
            exit 1
        fi

        if [ "$status" -ne 0 ]; then
            echo "  $AGENT exited with status $status, see $log (will retry on rerun)"
        else
            echo "  visualisation region is still empty, see $log (will retry on rerun)"
        fi
        break
    done

    sleep "$SLEEP_SECONDS"
done
