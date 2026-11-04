"""Call either unchanged CLI entry with the prepared offline service configuration."""
import argparse
import json
import os
from pathlib import Path
import sys

from serve.offline import environment, ROOT
from submission.assets import validate


def task_environment(deployment, state):
    data=json.loads(Path(deployment).read_text(encoding='utf-8-sig'))
    config=validate(data,verify_hashes=False)
    config_path=Path(state).resolve()/'model.json'
    if json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Running service configuration does not match deployment manifest')
    return environment(data,config_path,Path(state).resolve())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',default=os.environ.get('DEPLOYMENT_FILE'),required=not bool(os.environ.get('DEPLOYMENT_FILE')))
    parser.add_argument('--state-dir',default='/tmp/hls-agent-service')
    parser.add_argument('mode',choices=('agent','baseline'));parser.add_argument('input');parser.add_argument('output')
    args=parser.parse_args(argv)
    if sys.platform!='linux': parser.error('Native Linux required')
    try:
        env=task_environment(args.deployment,args.state_dir)
        script=ROOT/('run_baseline.sh' if args.mode=='baseline' else 'run.sh')
        os.execvpe('bash',['bash',str(script),args.input,args.output],env)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(2,'Task entry failed: '+str(exc)+'\n')


if __name__=='__main__': main()
