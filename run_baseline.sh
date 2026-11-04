#!/usr/bin/env bash
# 基线入口：run_baseline.sh <task_dir> <out_dir>
#
# 调用赛事方提供的 baseline.py，本文件与 baseline.py 均不得修改。
# 基线口径见《评分细则》。
exec python3 "$(dirname "$0")/baseline.py" "$1" "$2" "${TRACK:-hls}"
