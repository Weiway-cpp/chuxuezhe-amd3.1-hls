"""Supervise already-installed local Ollama, GPU telemetry and the competition API."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener,ProxyHandler

from submission.assets import validate

ROOT=Path(__file__).resolve().parents[1]


def environment(data, config_path, state):
    config=validate(data,verify_hashes=False)
    env=dict(os.environ)
    # ROCR selects the physical device; do not apply a second inherited mask to
    # the renumbered visible device set.
    for name in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'): env.pop(name,None)
    env.update(AGENT_CONFIG=str(config_path),MODEL_NAME=config['model'],LLM_BASE_URL=config['base_url'],
               OLLAMA_MODELS=data['models_root'],OLLAMA_HOST='127.0.0.1:11434',
               OLLAMA_CONTEXT_LENGTH=str(config['context_tokens']),OLLAMA_NUM_PARALLEL='1',
               OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_KEEP_ALIVE='-1',OLLAMA_NO_CLOUD='1',
               ROCR_VISIBLE_DEVICES=str(data['gpu_index']),
               HEALTH_FILE=str(state/'health.json'),HLS_TOOL_VERSION='2025.2',
               PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    return env


def stop(process):
    # The session leader may already have exited while an inference worker
    # remains in its process group. Always clean up the owned group.
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
    finally:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    process.wait(timeout=5)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',required=True)
    parser.add_argument('--state-dir',default='/tmp/hls-agent-service')
    args=parser.parse_args(argv)
    if sys.platform!='linux': parser.error('Native Linux is required for the offline service')
    children=[]; logs=[]; stopped=False
    def shutdown(*_):
        nonlocal stopped
        stopped=True
    previous={sig:signal.signal(sig,shutdown) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        data=json.loads(Path(args.deployment).read_text(encoding='utf-8-sig'))
        config=validate(data)
        token=os.environ.get('FPGACHINA_TOKEN','')
        if not token or any(ord(c)<33 or ord(c)>126 for c in token):
            raise ValueError('FPGACHINA_TOKEN must be a nonempty printable ASCII token')
        for port in (11434,7860):
            with socket.socket() as probe:
                probe.settimeout(.5)
                if probe.connect_ex(('127.0.0.1',port))==0:
                    raise RuntimeError('Required loopback service port is already occupied: '+str(port))
        state=Path(args.state_dir).resolve(); state.mkdir(parents=True,exist_ok=True)
        config_path=state/'model.json'; config_path.write_text(json.dumps(config),encoding='utf-8')
        (state/'health.json').unlink(missing_ok=True)
        env=environment(data,config_path,state)
        def launch(name,command):
            stream=(state/(name+'.log')).open('a',encoding='utf-8'); logs.append(stream)
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(process); return process
        backend=launch('ollama',[str(Path(data['runtime_root'])/'bin/ollama'),'serve'])
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and not stopped:
            if backend.poll() is not None: raise RuntimeError('Ollama exited; inspect local ollama.log')
            try:
                with build_opener(ProxyHandler({})).open('http://127.0.0.1:11434/api/version',timeout=1) as response:
                    if response.status==200: break
            except OSError: time.sleep(.2)
        else: raise RuntimeError('Local backend did not become available')
        if backend.poll() is not None: raise RuntimeError('New Ollama process did not own the backend service')
        launch('telemetry',[sys.executable,'-m','serve.gpu_telemetry','--pci-bdf',data['pci_bdf'],
                            '--health-file',str(state/'health.json'),'--log-file',str(state/'gpu.jsonl')])
        launch('api',[sys.executable,'-m','serve.competition_api','--port','7860'])
        print('Services started; /v1/health remains unready until real warmup and GPU telemetry pass.',flush=True)
        while not stopped:
            if any(p.poll() is not None for p in children): raise RuntimeError('A service exited; inspect state directory logs')
            time.sleep(.2)
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        print('Offline startup failed: '+str(exc),file=sys.stderr); return 1
    finally:
        for process in reversed(children): stop(process)
        for stream in logs: stream.close()
        for sig,handler in previous.items(): signal.signal(sig,handler)


if __name__=='__main__': raise SystemExit(main())
