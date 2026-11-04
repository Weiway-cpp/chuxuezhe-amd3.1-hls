# 方案包与离线部署

我们按 AMD 指南 3.1.5 整理方案包：Dockerfile、model/MODEL.md、agent/、skill/、serve/、run.sh、run_baseline.sh、REPORT.md 均具备。当前 ZIP 不包含官方基础镜像、Linux Ollama 二进制及模型权重，不能仅凭文件齐全宣称离线验收通过。我们采用以下离线部署流程，只使用预先准备的本地资产，保留官方基线原样。

## 在实验室准备一次

选择已验证的 Linux ROCm Ollama 发行目录（内含 bin/ollama 和其 lib/ 等运行依赖），保留实际使用的 Ollama 模型目录（内含 manifests/、blobs/）。Windows ollama.exe 不可放入 Linux 容器。官方基础镜像名称/摘要须由赛事方提供，我们尚未登记该信息；基础镜像至少应有 python3、bash 和 Vitis 2025.2，缺少第三方运行依赖时，应在冻结前补齐并记录版本。

在方案根目录运行，替换参数为实际路径/摘要/GPU 映射：

```bash
python3 -m submission.assets prepare \
  --runtime-root /opt/offline-ollama --models-root /opt/offline-models \
  --official-base 'ORGANIZER_IMAGE@sha256:ACTUAL_DIGEST' \
  --config config/model.ollama.example.json --gpu-index 0 --pci-bdf 0000:03:00.0 \
  --output submission/deployment.json
python3 -m submission.assets check --deployment submission/deployment.json
```

资产清单绑定整个运行时目录和指定模型 tag 实际引用的全部 blobs。此检查核验已有文件的哈希，不认证镜像来自官方、不证明 ROCm 兼容、不测显存。修改权重、运行时或模型 tag 后须重新准备，工具不会覆盖已有清单。确认 gpu-index 与 pci-bdf 确实对应同一块推理卡。

## 构建与启动

```bash
export OFFICIAL_BASE='ORGANIZER_IMAGE@sha256:ACTUAL_DIGEST'
export IMAGE_TAG='hls-agent:submission'
bash submission/build.sh
export FPGACHINA_TOKEN='TOKEN_FROM_ORGANIZER'
docker run --rm --name hls-agent --network none \
  --device /dev/kfd --device /dev/dri --group-add video \
  --mount type=bind,src=/opt/offline-ollama,dst=/opt/offline-ollama,readonly \
  --mount type=bind,src=/opt/offline-models,dst=/opt/offline-models,readonly \
  --env FPGACHINA_TOKEN \
  --env DEPLOYMENT_FILE=/opt/hls-agent/submission/deployment.json \
  "$IMAGE_TAG"
```

挂载目标路径必须与 deployment.json 的绝对路径一致。Linux Ollama 发行目录须包含与目标 gfx1201 GPU/ROCm 相容的依赖；脚本不会联网安装或静默回退成远程模型。容器无网络时，可在容器内部通过环回端点检查服务。需要对外投题的验证平台按赛事网络/隧道要求运行，禁止将模型推理切到外部服务。

默认启动统一的本地 Ollama、GPU 遥测和 127.0.0.1:7860 正式 API。任何子服务退出会停止整组服务。API 在模型预热与真实遥测通过前保持 unready；所有日志位于容器 /tmp/hls-agent-service。运行时哈希校验会读取大权重，需要给冷启动留足时间。连续请求、长请求显存峰值、物理 GPU 绑定、EDA license 与端到端验证仍须实机完成。

## 直接运行两条命令行入口

API 已统一配置子任务的模型环境。容器内的统一命令行包装器会读取同一 deployment.json 并核对正在使用的配置，然后调用原版入口：

```bash
bash serve/task.sh agent /path/to/task /path/to/agent-output
bash serve/task.sh baseline /path/to/task /path/to/baseline-output
```

Docker 默认环境将两条原版入口都指向本机 11434，并使用默认 qwen2.5-coder-32k:latest。若部署使用其他 tag，必须用上述包装器，或在每次 docker exec 时显式传入 `-e MODEL_NAME=实际tag`。若自行直接调用原版脚本，显式使用同一环境：

```bash
export AGENT_CONFIG=/tmp/hls-agent-service/model.json
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export MODEL_NAME=qwen2.5-coder-32k:latest
bash run.sh /path/to/task /path/to/agent-output
bash run_baseline.sh /path/to/task /path/to/baseline-output
```

MODEL_NAME 必须等于 deployment.json 的 model_config.model；如采用其他 tag，两者同时修改。两条入口仍调用同一服务/模型上下文，官方 baseline 文件未修改。保持首次生成对照记录、源文件哈希和模型声明一致。

## 提交前验收依据

将最终镜像摘要、deployment.json、模型声明、同模型成对评测、断网启动与 solve/health 记录、连续请求记录、整卡显存采样及权重/运行时许可随材料登记。资产检查输出始终标 deployment_verified=false；只有这些实际证据齐全后才能人工更新提交状态。我们将源码方案包与已验收镜像分别登记，当前交付范围为前者。
