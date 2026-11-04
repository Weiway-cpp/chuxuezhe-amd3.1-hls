# Use a preloaded organizer image by digest. Runtime/model assets are mounted.
# No apt, pip, model download or network install occurs during this build.
ARG OFFICIAL_BASE
FROM ${OFFICIAL_BASE}
WORKDIR /opt/hls-agent
COPY agent/ agent/
COPY scripts/ scripts/
COPY skill/ skill/
COPY model/ model/
COPY serve/ serve/
COPY submission/ submission/
COPY config/model.example.json config/model.ollama.example.json config/
COPY run.py run.sh run_baseline.sh baseline.py manifest.json REPORT.md NOTICE.md ./
RUN python3 --version && chmod +x run.sh run_baseline.sh serve/start.sh serve/task.sh submission/build.sh
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 HLS_TOOL_VERSION=2025.2
# Direct docker exec entry points share defaults; custom deployments use task.sh.
ENV LLM_BASE_URL=http://127.0.0.1:11434/v1 MODEL_NAME=qwen2.5-coder-32k:latest AGENT_CONFIG=/tmp/hls-agent-service/model.json
# DEPLOYMENT_FILE and FPGACHINA_TOKEN are supplied at runtime; missing assets fail.
CMD ["/bin/bash", "serve/start.sh"]
