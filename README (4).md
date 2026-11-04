# HLS 本地代码生成与反馈修复系统

v45 · 2026-11-04 · AMD Vitis HLS 2025.2

本目录是 v45 的提交源码方案包，采用单一英文根目录 chuxuezhe-agent。运行代码、启动脚本、配置模板、Dockerfile 和四份技能提示词保留 Project-agent-v45.zip 中的原始字节。本次统一了版本说明、设计报告与材料目录。

| 文件 | 内容 |
| --- | --- |
| [REPORT.md](REPORT.md) | 系统设计、v45 改动、已有结果及验证范围 |
| [SUBMISSION.md](SUBMISSION.md) | 离线资产准备、容器构建和服务启动 |
| [model/MODEL.md](model/MODEL.md) | 模型、上下文与部署声明 |
| [skill/README.md](skill/README.md) | 技能组织与适用范围 |
| [docs/verification.md](docs/verification.md) | 来源、文件一致性与打包检查 |
| [NOTICE.md](NOTICE.md) | 来源与许可 |
| [docs/ai-assistance.md](docs/ai-assistance.md) | 大模型辅助开发记录 |

正式单题入口为 run.sh，基线入口为 run_baseline.sh；两者使用同一本地模型服务。源文件检查命令为：

```bash
python3 -m scripts.source_manifest --check
```

本包不包含开发题库、完整单元测试、历史实验目录和其他工程 ZIP。开发资产保留在原始 Project-agent-v45.zip，正式求解不依赖这些题库和参考答案。

模型权重、Linux 推理运行时和官方基础镜像摘要未随输入提供，离线容器尚未验收；ready_for_submission=false 保留这一实际状态。已有源码和部署脚本不等于已经交付可直接加载的离线镜像。
