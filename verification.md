# 提交包核验

来源是 Project-agent-v45.zip。本次核验其 SOURCE_MANIFEST.json 和 UPSTREAM_MANIFEST.json，保留正式入口需要的运行模块、配置、脚本、Dockerfile 与技能文件；源码内容没有改写。完整开发题库、测试和历史文件仍在原输入 ZIP。

PACKAGING.json 记录来源指纹、运行文件哈希、对 v44 的代码文件差异和检查范围；VALIDATION.json 只声明本次实际完成的打包检查。原验证声明仍为 v39，保存在 docs/evidence/source-validation.json；原 manifest 及版本材料也留作来源记录。

我们检查 Python 语法与模块导入、模板配置、CLI 帮助入口、Docker COPY 源路径、Markdown 本地链接、Shell LF 换行及 ZIP 可执行权限。正式基线与输入的固定上游副本逐字节一致。ZIP 仅有 chuxuezhe-agent/ 一个根目录，没有嵌套工程 ZIP、运行日志、缓存或本地秘密配置。

在解包后的 chuxuezhe-agent 目录执行：

```bash
python3 -m scripts.source_manifest --check
```

发布时另做 ZIP CRC、内容逐字节比较与独立解包后的清单核验，结果记录在 ZIP 同目录的 inspection.json。没有运行完整单元测试、真实模型、Vitis 或 Docker；模型资产与离线环境尚未验收，ready_for_submission 保持 false。
