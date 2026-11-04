#!/usr/bin/env python3
"""
基线运行器。赛事方提供，队伍原样带上，不得修改。

用法：  baseline.py <task_dir> <out_dir> [rtl|hls]

    产出   <out_dir>/solution.v   或  solution.cpp
           <out_dir>/trace.jsonl

在无智能体、无技能包、单次直接生成的条件下解一道题，作为增益项的分母。
prompt 与采样参数由赛事方固定，队伍决定模型、量化、推理后端与服务配置。

对队伍的要求：推理服务须暴露 OpenAI 兼容的 `/v1/chat/completions` 端点，
默认 `http://127.0.0.1:8000/v1`，可用 `LLM_BASE_URL` 覆盖地址。使用
chat/completions 而非 completions，以便模型自带的 chat template 在服务端生效。

trace.jsonl 首行记录本脚本的 SHA-256 与服务端报出的模型名，供赛事方核验。
基线的 trace 须只有一次 llm 调用、零次工具调用。
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ── 以下常量构成基线口径，不要改 ────────────────────────────────────────────
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 8192
TIMEOUT_S = 300
# MAX_TOKENS 取 8192：混合推理模型会先产出思考 token，预算过小会截断输出。
# 请求仅使用 OpenAI 标准字段，不含厂商专有参数。

SYS = {
    "rtl": ("You are an expert Verilog designer. Reply with a single synthesizable "
            "SystemVerilog module named TopModule. Output only code — no prose, no "
            "markdown fences."),
    "hls": ("You are an expert Vitis HLS engineer. Reply with a single C++ source "
            "file implementing the requested top-level function. Include the header "
            "shown in the interface section. Use only synthesizable C++ — no "
            "dynamic allocation, no recursion, no system calls. Output only code — "
            "no prose, no markdown fences."),
}
# ── 常量到此为止 ──────────────────────────────────────────────────────────

OUT_NAME = {"rtl": "solution.v", "hls": "solution.cpp"}
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def self_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def post_json(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def served_model():
    """问服务端它实际在服务什么模型。用于与 MODEL.md 交叉核对。"""
    try:
        req = urllib.request.Request(BASE_URL + "/models")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read()).get("data", [])
        return data[0]["id"] if data else None
    except Exception:
        return None


def extract(text, track):
    """从回复中提取代码。模型常附加 markdown 围栏或说明文字。

    与 agent.py 使用同一套抽取逻辑，以保证分子分母口径一致。
    """
    m = re.search(r"```(?:systemverilog|verilog|sv|cpp|c\+\+|c)?\s*(.*?)```",
                  text, re.S)
    if m:
        text = m.group(1)
    if track == "rtl":
        m = re.search(r"(module\s+TopModule\b.*?endmodule)", text, re.S)
        return (m.group(1) if m else text).strip() + "\n"
    return text.strip() + "\n"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.strip().splitlines()[2])
    task_dir, out_dir = sys.argv[1], sys.argv[2]
    track = (sys.argv[3] if len(sys.argv) > 3
             else os.environ.get("TRACK", "rtl")).lower()
    if track not in SYS:
        sys.exit(f"未知 track：{track}")
    os.makedirs(out_dir, exist_ok=True)

    prompt = open(os.path.join(task_dir, "prompt.txt")).read()
    iface_p = os.path.join(task_dir, "interface.txt")
    interface = open(iface_p).read() if os.path.isfile(iface_p) else ""
    if interface:
        prompt += "\n\nInterface:\n" + interface

    trace = [{"tool": "baseline_meta",
              "script_sha256": self_sha256(),
              "served_model": served_model(),
              "temperature": TEMPERATURE, "top_p": TOP_P,
              "max_tokens": MAX_TOKENS, "track": track}]

    code, err = "", None
    t0 = time.time()
    try:
        d = post_json(BASE_URL + "/chat/completions", {
            # model 留空，由服务端选择默认模型
            "model": os.environ.get("MODEL_NAME") or (served_model() or "default"),
            "messages": [{"role": "system", "content": SYS[track]},
                         {"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        })
        choice = d["choices"][0]
        usage = d.get("usage", {})
        # 推理模型（gpt-oss 一类）把思考过程放在 message.reasoning 里，思考 token
        # 同样占 max_tokens。预算耗尽时 content 直接是 null，而不是空串。
        # 实测：Prob057_kmap2 在 8192 与 16000 下都是 finish_reason=length、
        # content=None —— 加大预算只会让它思考更久，不是配置问题。
        reply = choice["message"].get("content") or ""
        # ★ 先落 trace 再解析。这行原本在 extract() 之后，而 extract(None) 会抛
        #   TypeError，于是最需要诊断信息的那 23 个样本里，finish_reason 恰好没记下来，
        #   失败原因被写成一个看不出所以然的 TypeError。
        trace.append({"ts": round(t0, 3), "tool": "llm", "round": 0,
                      "tokens_in": usage.get("prompt_tokens"),
                      "tokens_out": usage.get("completion_tokens"),
                      "finish": choice.get("finish_reason"),
                      "empty_content": not reply,
                      "sec": round(time.time() - t0, 2)})
        code = extract(reply, track)
    except Exception as e:                       # 超时、连不上、返回格式不对
        err = f"{type(e).__name__}: {e}"
        trace.append({"ts": round(t0, 3), "tool": "llm", "round": 0,
                      "error": err[:300], "sec": round(time.time() - t0, 2)})

    # 失败时写空产物，判定器按 L0 处理。
    with open(os.path.join(out_dir, OUT_NAME[track]), "w") as f:
        f.write(code)
    with open(os.path.join(out_dir, "trace.jsonl"), "w") as f:
        for e in trace:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if err:
        print(f"baseline: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
