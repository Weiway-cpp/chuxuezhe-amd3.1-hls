"""Bind existing Linux runtime/model assets; never download or claim deployment readiness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from agent.model import load_config, validate_config


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def inside(root, name):
    if not isinstance(name,str) or not name or '\\' in name:
        raise ValueError('Asset path must be a relative POSIX path')
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing asset or path outside its declared root')
    return path


def inventory(runtime_root, models_root, model):
    runtime=Path(runtime_root).resolve(); models=Path(models_root).resolve()
    executable=inside(runtime,'bin/ollama')
    if os.name=='posix' and not os.access(executable,os.X_OK):
        raise ValueError('Linux Ollama binary is not executable')
    with executable.open('rb') as stream:
        if stream.read(4)!=b'\x7fELF': raise ValueError('Ollama runtime must be a Linux ELF build')
    if not re.fullmatch(r'[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?',model):
        raise ValueError('Use one local Ollama library tag name, without registry/path syntax')
    name,_,tag=model.partition(':'); tag=tag or 'latest'
    manifest_name='manifests/registry.ollama.ai/library/'+name+'/'+tag
    manifest=inside(models,manifest_name)
    data=json.loads(manifest.read_text(encoding='utf-8'))
    descriptors=[data['config'],*data['layers']]
    model_names={manifest_name}
    for descriptor in descriptors:
        value=descriptor['digest']
        if not re.fullmatch(r'sha256:[0-9a-f]{64}',value): raise ValueError('Invalid Ollama blob digest')
        relative='blobs/'+value.replace(':','-')
        path=inside(models,relative)
        if digest(path)!=value.split(':')[1]: raise ValueError('Ollama blob hash mismatch')
        model_names.add(relative)
    runtime_names=[p.relative_to(runtime).as_posix() for p in runtime.rglob('*') if p.is_file()]
    return {'runtime':{n:digest(inside(runtime,n)) for n in sorted(runtime_names)},
            'models':{n:digest(inside(models,n)) for n in sorted(model_names)}}


def validate(data, *, verify_hashes=True):
    if not isinstance(data,dict) or data.get('schema_version')!=1:
        raise ValueError('Unsupported deployment asset schema')
    if not re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}',data.get('official_base','')):
        raise ValueError('Provide the organizer base image by exact sha256 digest')
    if type(data.get('gpu_index')) is not int or data['gpu_index']<0:
        raise ValueError('One nonnegative GPU index is required')
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',data.get('pci_bdf','')):
        raise ValueError('Explicit physical GPU PCI BDF is required')
    config=validate_config(data['model_config'])
    if config['base_url']!='http://127.0.0.1:11434/v1':
        raise ValueError('Offline inference endpoint must be the loopback Ollama service')
    if verify_hashes:
        actual=inventory(data['runtime_root'],data['models_root'],config['model'])
        if actual!=data.get('asset_sha256'): raise ValueError('Deployment assets changed after preparation')
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('runtime-root','models-root','official-base','config','pci-bdf','output'):
        prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--gpu-index',type=int,required=True)
    check=sub.add_parser('check'); check.add_argument('--deployment',required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='prepare':
            config=load_config(args.config)
            config['base_url']='http://127.0.0.1:11434/v1'
            data={'schema_version':1,'official_base':args.official_base,
                  'runtime_root':str(Path(args.runtime_root).resolve()),
                  'models_root':str(Path(args.models_root).resolve()),'model_config':config,
                  'gpu_index':args.gpu_index,'pci_bdf':args.pci_bdf,
                  'ready_for_submission':False,
                  'scope':'Asset binding only; image provenance, GPU mapping and end-to-end runtime unverified'}
            validate(data,verify_hashes=False)
            data['asset_sha256']=inventory(data['runtime_root'],data['models_root'],config['model'])
            with Path(args.output).open('x',encoding='utf-8') as stream:
                json.dump(data,stream,ensure_ascii=False,indent=2)
        else:
            data=json.loads(Path(args.deployment).read_text(encoding='utf-8-sig')); validate(data)
        print(json.dumps({'assets_valid':True,'deployment_verified':False,'ready_for_submission':False}))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(2,'Asset check failed: '+str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
