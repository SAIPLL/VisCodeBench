# VisCodeBench Tutorial

This guide takes you from a fresh clone of this repository to a submission on the [Codabench leaderboard](https://www.codabench.org/competitions/16125/). It uses a coding agent, either [OpenAI Codex CLI](https://github.com/openai/codex) or [Claude Code](https://code.claude.com/docs), running inside a Docker container to generate the predictions.

1. [Set up](#1-set-up)
2. [Download the data](#2-download-the-data)
3. [Build the Docker image](#3-build-the-docker-image)
4. [Start an agent container and log in](#4-start-an-agent-container-and-log-in)
5. [Generate predictions](#5-generate-predictions)
6. [Log the displayed data](#6-log-the-displayed-data)
7. [Submit](#7-submit)

Appendices: [Prompt template](#a-prompt-template) · [Running without Docker](#b-running-without-docker)

**What gets submitted:** one `<sample_id>.jsonl` log per sample, recording the data your visualisation actually plots. You produce it locally by running each completed script under the `t2v_eval` logging hooks. Codabench only scores these logs and never runs your code, so do not upload Python files, data files or images.

**Why Docker:** the agent writes files and runs code without asking first, and its prompts are built from third-party data. The container keeps it away from the rest of your machine. It only sees the data (read-only), the run script, and the folder for its own run, so it can't read your files, other models' answers, or modify the original data.

---

## 1. Set up

You need [Git](https://git-scm.com/downloads), [Docker](https://docs.docker.com/get-started/get-docker/) (Docker Desktop on macOS/Windows, or Docker Engine on Linux), and an account for the agent you want to use: a ChatGPT account or OpenAI API key for Codex, or a Claude subscription or Anthropic API key for Claude Code.

```bash
git clone https://github.com/SAIPLL/VisCodeBench.git
cd VisCodeBench
```

All commands below are run from the repository root unless they say "inside the container".

---

## 2. Download the data

Download `viscodebench-public-data.zip` from the Resources section of the [competition page](https://www.codabench.org/competitions/16125/) and extract it into `data/codabench`:

```bash
mkdir -p data/codabench
unzip viscodebench-public-data.zip -d data/codabench
```

You should now have four subset folders, with one folder per sample (2,659 samples in total):

```text
data/codabench
├── arxiv-t2v-easy     (455 samples)
│   ├── r1_mail_001_1_1
│   │   ├── model_accuracy_data.json
│   │   ├── outputs/
│   │   ├── r1_mail_001_1_1.py
│   │   └── r1_mail_001_1_1.vis-request.txt
│   └── ...
├── arxiv-t2v-hard     (343 samples)
├── github-t2v         (837 samples)
└── owid-t2v           (1,024 samples)
```

Each sample folder contains:

```text
data/codabench/<subset>/<sample_id>/
├── <sample_id>.py               # context code; the prediction goes here
├── <sample_id>.vis-request.txt  # natural-language visualisation request
├── outputs/                     # generated images are saved here
└── ...                          # optional data files
```

The prediction is the code written into `<sample_id>.py` between these markers, which are already at the end of the file. Keep the rest of the file unchanged:

```python
##START VISUALISATION CODE
#TODO
##END VISUALISATION CODE
```

`data/` is gitignored, so the data never ends up in commits.

---

## 3. Build the Docker image

The [`Dockerfile`](../Dockerfile) installs:

- Python 3.12.
- The `t2v_eval` package, with the plotting and scientific libraries the samples use.
- Both coding agents, Codex CLI and Claude Code, at pinned versions.

It also creates a non-root user matching yours, so files the container writes stay owned by you.

```bash
docker build -t viscodebench \
  --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
```

The first build takes several minutes and the image is a few GB, mostly because of the scientific Python libraries (PyTorch is installed as CPU-only to keep it smaller). To use other agent versions, add `--build-arg CODEX_VERSION=<version>` or `--build-arg CLAUDE_CODE_VERSION=<version>`.

> **Build fails with `unable to get local issuer certificate`?** Your network intercepts HTTPS (common on university and corporate networks, e.g. Zscaler). Put the network's root certificate in `docker/certs/` and rebuild; see [docker/certs/README.md](../docker/certs/README.md).

---

## 4. Start an agent container and log in

Each run of an agent with a given model gets its own container. Choose the agent and model on your machine, create the run folder, and start the container:

```bash
AGENT=codex  MODEL=gpt-6.1-sol        # or: AGENT=claude MODEL=claude-opus-5-5
mkdir -p "runs/$AGENT/$MODEL"

docker run -it --rm \
  -v "$PWD/data/codabench":/work/data/codabench:ro \
  -v "$PWD/run_agent.sh":/work/run_agent.sh:ro \
  -v "$PWD/runs/$AGENT/$MODEL":"/work/runs/$AGENT/$MODEL" \
  -v viscodebench-codex-home:/home/agent/.codex \
  -v viscodebench-claude-home:/home/agent/.claude \
  -e AGENT="$AGENT" -e MODEL="$MODEL" -e RUN_DIR="runs/$AGENT/$MODEL" \
  viscodebench
```

The container sees only what this run needs:

- **`data/codabench`, read-only (`:ro`).** The agent can read the samples but can't change the originals.
- **`run_agent.sh`, read-only.**
- **`runs/$AGENT/$MODEL`, read-write.** This is where this run's answers are written, so they stay on your machine after the container exits.
- **Nothing else.** Your other files, other runs' answers, and the logs and submissions from step 6 are all invisible inside the container.

The two `viscodebench-*-home` volumes keep each agent's login and settings, so you only log in once. Your host's own credentials are not shared with the container. `--rm` deletes the container when you exit; start a new one with the same command at any time.

Inside the container, log in to the agent you will use. You only need one of them.

### Codex

With a ChatGPT account, use the device-code flow. Open the link it prints in a browser on your machine and enter the code:

```bash
codex login --device-auth
```

Or with an API key:

```bash
printenv OPENAI_API_KEY | codex login --with-api-key   # after: export OPENAI_API_KEY=...
```

Check with `codex login status`.

### Claude Code

With a Claude subscription, start the login. It prints a URL: open it in a browser on your machine, sign in, and paste the code it shows back into the terminal:

```bash
claude auth login
```

Or with an Anthropic API key, pass it when starting the container by adding `-e ANTHROPIC_API_KEY` to the `docker run` command above, after setting `export ANTHROPIC_API_KEY=...` on your machine. In headless mode, Claude Code uses the key whenever it is set.

Check with `claude auth status`.

---

## 5. Generate predictions

Inside the container from step 4, run [`run_agent.sh`](../run_agent.sh). `AGENT`, `MODEL` and `RUN_DIR` are already set by the `docker run` command. For each sample, it copies the sample into a folder for this run, sends the request to the agent with the [prompt template](#a-prompt-template), and the agent fills in the visualisation region of that copy's `<sample_id>.py`. `data/codabench` itself is never modified. Start with a few samples and check the results before running everything:

```bash
# inside the container
MAX_SAMPLES=5 bash run_agent.sh
```

Then run the rest with `bash run_agent.sh`. The script skips samples that already finished, so you can stop it with Ctrl+C and rerun it at any time.

Settings, passed as environment variables (all are described at the top of the script):

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENT` | (required) | `codex` or `claude`. |
| `MODEL` | agent's default | Model name passed to the agent's `--model`. |
| `MAX_SAMPLES` | `0` (all) | Process at most this many samples, then stop. Already-done samples don't count. |
| `SUBSETS` | all | Space-separated subsets, e.g. `SUBSETS="github-t2v owid-t2v"`. |
| `DATA_DIR` | `data/codabench` | Folder containing the subset folders. |
| `RUN_DIR` | `runs/<AGENT>/<MODEL>` | This run's copies of the samples, with the agents' answers and logs. Without `MODEL`, the folder is `runs/<AGENT>/default`. |
| `ON_USAGE_LIMIT` | `wait` | What to do when the agent hits a usage or rate limit: `wait` or `stop`. |
| `USAGE_LIMIT_WAIT` | `1800` | Seconds to wait before retrying after a limit. |
| `SLEEP_SECONDS` | `5` | Pause between samples. |

### Where the answers go

Each agent and model gets its own folder, with the same layout as `data/codabench`:

```text
runs/codex/gpt-6.1-sol/
├── arxiv-t2v-easy/
│   └── r1_mail_001_1_1/
│       ├── r1_mail_001_1_1.py      # the agent's answer
│       ├── outputs/output_1.png    # the agent's chart
│       ├── model_accuracy_data.json
│       └── .run/                   # codex.log, elapsed_seconds.txt, done
├── arxiv-t2v-hard/
├── github-t2v/
└── owid-t2v/
```

Each agent and model starts every sample from a fresh copy of the untouched original. Its container from step 4 only contains its own run folder, so it cannot see other runs' answers. To compare models, start a new container with a different `AGENT` or `MODEL`. Copies are made only for the samples you run, so the folder grows as you go.

A sample is only marked done (`.run/done`) if the agent succeeded and the visualisation region is no longer empty. To redo a sample from scratch, delete its folder, for example `runs/codex/gpt-6.1-sol/arxiv-t2v-easy/r1_mail_001_1_1/`.

The script also checks that each original in `data/codabench` still has an empty visualisation region. If one was edited (for example, by an older version of this script), the script skips that sample and asks you to restore the original from `viscodebench-public-data.zip`.

### Usage and rate limits

Running all 2,659 samples will likely hit your plan's usage limits (Codex's or Claude's 5-hour and weekly limits) or the API's rate limits. Both agents already retry brief rate-limit errors themselves. When an agent gives up and reports a limit, the script:

- **`ON_USAGE_LIMIT=wait` (default):** prints the agent's message (for example, `You've hit your session limit · resets 3:45pm`) and sleeps for `USAGE_LIMIT_WAIT` seconds. Then it retries the **same** sample, and keeps doing so until the limit resets. Leave it running overnight, or press Ctrl+C at any time.
- **`ON_USAGE_LIMIT=stop`:** exits with status 2. Rerun the same command later to continue from that sample.

If the agent reports a billing problem (`Credit balance is too low`, a spend limit, or an exhausted API quota), the script stops instead, because waiting won't fix it.

### Safety inside the container

Inside the container, the agents' own sandboxes and permission prompts are turned off (`AGENT_FULL_ACCESS=1` is set in the image). This means `codex --sandbox danger-full-access` and `claude --dangerously-skip-permissions`; the container is the sandbox. Only run the script this way inside the container, never directly on your machine.

The agent still has internet access (it needs it to reach its model). It can also read its own login in the `viscodebench-*-home` volumes. Use a dedicated account or API key if that matters to you.

The script is a reference to read and adapt. You can swap in another agent or your own system, as long as it fills in the region between the markers.

---

## 6. Log the displayed data

Exit the agent container. Then, on your machine, run every completed script under the `t2v_eval` logging hooks and build the submission. This uses a separate container with no agent in it, which can see the whole repository:

```bash
RUN=codex/gpt-6.1-sol   # the agent/model folder under runs/ to submit
docker run --rm -v "$PWD":/work viscodebench \
  python -m t2v_eval.batch_log \
    --data_dir runs/$RUN \
    --log_dir logs/$RUN \
    --submission_dir submission/$RUN \
    --workers 4 --zip
```

Use separate `--log_dir` and `--submission_dir` folders for each agent and model, as above, so their logs and submissions don't overwrite each other.

For each sample, it:

1. Copies `<sample_id>.py` to a temporary file and inserts the logging line just before `##START VISUALISATION CODE`. This is the same way the reference logs were produced.
2. Runs the copy from the sample folder, so relative data paths resolve, with a non-interactive matplotlib backend and a per-sample timeout.
3. Writes the raw log to `logs/$RUN/<subset>/<sample_id>/series.jsonl` and copies it to `submission/$RUN/<subset>/<sample_id>.jsonl`.

When it finishes, it prints a per-subset summary and writes `logs/$RUN/logging_report.csv`, listing every sample's state (`HAVING_DATA`, `NO_DATA` or `BAD_INPUT`) and the error for failed ones. A sample without a log scores 0, so check the report and fix what you can (for example, rerun those samples in step 5) before submitting.

Useful options (see `python -m t2v_eval.batch_log --help`):

| Option | Default | Purpose |
| --- | --- | --- |
| `--subsets` | all | Only process these subsets, e.g. `--subsets github-t2v owid-t2v`. |
| `--timeout` | `300` | Per-sample timeout in seconds. |
| `--workers` | `1` | Samples to run in parallel. |
| `--skip_existing` | off | Reuse logs from an earlier run that already have a `series.jsonl`. |
| `--zip` | off | Also write `<submission_dir>.zip`, ready to upload. |
| `--log_dir`, `--submission_dir` | `logs`, `submission` | Output locations. |

This step runs model-generated code too, which is why it also runs in a container.

---

## 7. Submit

The zip is now at `submission/$RUN.zip` on your machine (e.g. `submission/codex/gpt-6.1-sol.zip`). It has one `<sample_id>.jsonl` per sample, with the subset folders at the top level, as Codabench requires:

```text
gpt-6.1-sol.zip
├── arxiv-t2v-easy/
│   ├── r1_mail_001_1_1.jsonl
│   └── ...
├── arxiv-t2v-hard/
├── github-t2v/
└── owid-t2v/
```

The scorer uses these folders to route each prediction to its subset's leaderboard. If you didn't use `--zip`, create the archive from inside the submission folder, e.g. `cd submission/codex/gpt-6.1-sol && zip -r ../gpt-6.1-sol.zip .`. You can leave out subsets you have no predictions for; their samples score 0.

Upload the zip on the **My Submissions** page of the [competition](https://www.codabench.org/competitions/16125/), and put your model's name in the submission comment (for example, `Codex-GPT-5`). One submission is scored on all four subset leaderboards, using two metrics:

- **Displayed Data Similarity** (primary, from 0 to 1): how closely your plotted values, labels and structure match the reference.
- **Chart Type Accuracy**: the fraction of samples whose chart types cover all the reference's chart types.

Missing or malformed JSONL files score 0 on both metrics. Each leaderboard keeps your best submission, and you can resubmit at any time.

---

## A. Prompt template

This is the prompt [`run_agent.sh`](../run_agent.sh) sends to either agent for each sample. `{SAMPLE_ID}` is the sample folder name, `{VIS_REQUEST}` is the contents of `<sample_id>.vis-request.txt`, and `{PYTHON_PATH}` is the interpreter with the plotting libraries (`/usr/local/bin/python` in the container). If you use a different agent, use the same prompt so your results are comparable.

```text
Given data files and a code context file '{SAMPLE_ID}.py'.

Visualisation request:
'{VIS_REQUEST}'

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
(11) Execute the visualisation program using the Python interpreter at: '{PYTHON_PATH}'
```

---

## B. Running without Docker

You can also install `t2v_eval` directly and use your own Text-to-Vis system. You need Python 3.10–3.12:

```bash
conda create -n vcbench_env python=3.12
conda activate vcbench_env
pip install "git+https://github.com/SAIPLL/VisCodeBench.git"
python -c "import t2v_eval; print(t2v_eval.__version__)"
```

Then fill in each `<sample_id>.py` with your system, and run steps 6 and 7 as above (without the container). If you run a coding agent on your own machine, keep its sandbox and permission checks on. `run_agent.sh` does this by default outside the container (`AGENT_FULL_ACCESS=0`). Codex then runs with `--sandbox workspace-write`. Claude Code may edit files in the sample folder and run only the Python interpreter. Never use `danger-full-access` or `--dangerously-skip-permissions` outside a disposable container or VM.

On Linux, Codex's sandbox needs unprivileged user namespaces. If its commands fail with `bwrap: ... Operation not permitted`, your machine doesn't allow them; use the Docker setup above instead.
