"""Sample Linux AMDGPU sysfs memory counters for the competition health endpoint.

Uses real device-wide counters, not model-size estimates. Explicit PCI selection
is required: the operator must bind inference to this same GPU. Samples do not
prove process attribution, single-GPU use, or an unobserved transient peak.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from agent.competition import competition_config
from agent.model import ChatClient, ModelError
from serve.competition_api import probe_models

SYSFS = Path('/sys/bus/pci/devices')
GIB = 1024 ** 3
BDF = re.compile(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')


def probe_inference(config):
    # A listed model ID alone may be an unloaded, on-demand model. One bounded
    # real request at collector startup confirms the configured backend responds.
    response = ChatClient({**config, 'max_tokens':128, 'temperature':0}).generate(
        [{'role':'user', 'content':'Reply with OK.'}], timeout=30)
    return bool(response.get('content', '').strip())


def _integer(path):
    value = path.read_text(encoding='ascii').strip()
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError('Invalid GPU counter: ' + path.name)
    return int(value)


def read_device(pci_bdf, *, sysfs=SYSFS):
    if not isinstance(pci_bdf, str) or not BDF.fullmatch(pci_bdf):
        raise ValueError('Expected PCI address such as 0000:03:00.0')
    device = Path(sysfs) / pci_bdf
    if device.joinpath('vendor').read_text(encoding='ascii').strip().lower() != '0x1002':
        raise ValueError('Selected device is not AMD')
    total = _integer(device / 'mem_info_vram_total')
    used = _integer(device / 'mem_info_vram_used')
    if total <= 0 or not 0 <= used <= total:
        raise ValueError('Inconsistent GPU memory counters')
    busy = None
    try:
        busy = _integer(device / 'gpu_busy_percent')
        if not 0 <= busy <= 100:
            busy = None
    except (OSError, ValueError):
        pass
    return {'vram_used_bytes': used, 'vram_total_bytes': total,
            'vram_gb': used / GIB, 'gpu_busy_percent': busy}


def list_devices(*, sysfs=SYSFS):
    devices = []
    for path in sorted(Path(sysfs).glob('*')):
        try:
            devices.append({'pci_bdf': path.name, **read_device(path.name, sysfs=sysfs)})
        except (OSError, ValueError):
            continue
    return devices


class Collector:
    def __init__(self, config, pci_bdf, *, sysfs=SYSFS, model_probe=probe_models, inference_probe=probe_inference):
        parsed = urlsplit(config['base_url'])
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Telemetry requires a loopback HTTP model endpoint')
        if not config.get('model') or config['model'].startswith('REPLACE_'):
            raise ValueError('Configure the real served model before starting telemetry')
        if not BDF.fullmatch(pci_bdf):
            raise ValueError('Invalid PCI address')
        self.config, self.pci_bdf = dict(config), pci_bdf
        self.sysfs, self.model_probe = Path(sysfs), model_probe
        self.inference_probe = inference_probe
        self.inference_confirmed_at = None
        self.run_id = uuid.uuid4().hex
        self.peak = None
        self.samples = self.failed_samples = 0

    def warmup(self):
        self.inference_confirmed_at = None
        if not self.inference_probe(self.config):
            raise ModelError('The local model did not complete the startup inference check')
        self.inference_confirmed_at = time.time()

    def sample(self):
        self.samples += 1
        record = {'run_id': self.run_id, 'model': self.config['model'],
                  'pci_bdf': self.pci_bdf, 'model_loaded': False, 'vram_gb': None,
                  'measurement_source': 'linux_amdgpu_sysfs', 'memory_unit': 'GiB',
                  'scope': 'selected_device_total_usage_not_process_attribution',
                  'peak_scope': 'maximum_observed_sample_in_this_collector_run',
                  'sample_index': self.samples}
        record['inference_confirmed_at'] = self.inference_confirmed_at
        try:
            loaded = bool(self.model_probe(self.config))
            if not loaded:
                # A backend outage/restart invalidates the startup inference
                # evidence. Restart the collector after the model is ready.
                self.inference_confirmed_at = None
                record['inference_confirmed_at'] = None
            counters = read_device(self.pci_bdf, sysfs=self.sysfs)
            record.update(counters, model_loaded=loaded and self.inference_confirmed_at is not None)
            used = counters['vram_used_bytes']
            self.peak = used if self.peak is None else max(self.peak, used)
        except (OSError, ValueError, TypeError) as exc:
            self.failed_samples += 1
            record['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        record.update(observed_at=time.time(), failed_samples=self.failed_samples,
                      peak_observed_vram_bytes=self.peak)
        return record


def publish(record, health_file, log_file):
    """Keep JSONL evidence, then atomically replace the current health sample."""
    health_file, log_file = Path(health_file), Path(log_file)
    if health_file.resolve() == log_file.resolve():
        raise ValueError('Health and history paths must differ')
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
    health_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = health_file.with_name(health_file.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with log_file.open('a', encoding='utf-8') as history:
            history.write(serialized)
        temporary.write_text(serialized, encoding='utf-8')
        os.replace(temporary, health_file)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List readable AMD PCI devices and exit')
    parser.add_argument('--pci-bdf', help='Explicit physical PCI device used by inference')
    parser.add_argument('--health-file', default=os.environ.get('HEALTH_FILE', 'runs/health.json'))
    parser.add_argument('--log-file', default='runs/gpu_telemetry.jsonl')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--once', action='store_true', help='One measurement; no background service')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Requires native Linux with readable AMDGPU sysfs counters')
    if args.list:
        print(json.dumps(list_devices(), indent=2))
        return 0
    if not args.pci_bdf:
        parser.error('--pci-bdf is required; use --list and verify the inference GPU mapping')
    if not math.isfinite(args.interval) or not 0.25 <= args.interval <= 60:
        parser.error('--interval must be between 0.25 and 60 seconds')
    collector = Collector(competition_config(), args.pci_bdf)
    publish({'observed_at':time.time(), 'model':collector.config['model'],
             'model_loaded':False, 'vram_gb':None, 'event':'collector_initializing'},
            args.health_file, args.log_file)
    try:
        collector.warmup()
    except (OSError, ValueError, ModelError) as exc:
        print('Telemetry inference check failed: ' + str(exc), file=sys.stderr)
        return 1
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopped.is_set():
            record = collector.sample()
            publish(record, args.health_file, args.log_file)
            if args.once:
                print(json.dumps(record, indent=2))
                return int(record.get('error') is not None or not record['model_loaded'])
            stopped.wait(args.interval)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not args.once:
            publish({'observed_at': time.time(), 'model': collector.config['model'],
                     'run_id': collector.run_id, 'pci_bdf': collector.pci_bdf,
                     'model_loaded': False, 'vram_gb': None, 'event': 'collector_stopped'},
                    args.health_file, args.log_file)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
