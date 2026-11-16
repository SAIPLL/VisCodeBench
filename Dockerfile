# VisCodeBench runner image: t2v_eval + Codex CLI + Claude Code.
#
# The container is the isolation boundary for the coding agents: it only sees
# the folder you mount at /work, so the agents' own sandboxes and permission
# prompts are turned off inside it (AGENT_FULL_ACCESS=1). See docs/tutorial.md.
#
# Build (from the repo root):
#   docker build -t viscodebench \
#     --build-arg UID=$(id -u) --build-arg GID=$(id -g) .

ARG NODE_VERSION=22.12.0
ARG PYTHON_VERSION=3.12.8

FROM node:${NODE_VERSION}-bookworm-slim AS node

FROM python:${PYTHON_VERSION}-slim-bookworm

# --- Extra CA certificates (optional) ---
# On networks that intercept HTTPS (e.g. Zscaler), put the network's root CA
# as a PEM *.crt file in docker/certs/ before building; see docker/certs/README.md.
# On other networks, leave the folder as it is: no *.crt files means nothing is
# added and the standard public CAs are used. Nothing here needs to be removed.
# Every tool below (apt, npm, pip, Python, Codex) is pointed at the system store.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# --- Node.js (for the Codex CLI), copied from the official image ---
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# --- Small system tools the agent commonly uses ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl procps ripgrep zip unzip \
 && rm -rf /var/lib/apt/lists/*

# --- Coding agents: Codex CLI and Claude Code (pin the versions you tested) ---
# Installed from /tmp: Claude Code's installer can hang when run from /.
ARG CODEX_VERSION=0.166.0
ARG CLAUDE_CODE_VERSION=2.1.318
WORKDIR /tmp
RUN npm install -g "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
 && npm cache clean --force

# --- t2v_eval and the plotting/scientific libraries the samples need ---
# CPU-only PyTorch wheels keep the image several GB smaller than the CUDA ones.
# Dependencies are installed before copying the package source, so editing
# t2v_eval does not re-download them; the pip cache mount survives failed
# builds, so a retry after a network timeout resumes instead of restarting.
WORKDIR /opt/viscodebench
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY pyproject.toml README.md ./
COPY t2v_eval ./t2v_eval
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps . \
 && python -c "import t2v_eval; print('t2v_eval', t2v_eval.__version__)"

# --- Unprivileged user; match your host UID/GID so files written to the
#     mounted folder are owned by you, not root ---
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" agent 2>/dev/null || true \
 && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash agent \
 && mkdir -p /home/agent/.codex /home/agent/.claude /work \
 && chown -R "${UID}:${GID}" /home/agent /work

USER agent
WORKDIR /work

# Defaults for run_agent.sh inside the container: the container is the
# sandbox, so the agents' own sandboxes and permission prompts are turned off.
ENV AGENT_FULL_ACCESS=1 \
    PYTHON_BIN=/usr/local/bin/python \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    CLAUDE_CONFIG_DIR=/home/agent/.claude \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

CMD ["bash"]
