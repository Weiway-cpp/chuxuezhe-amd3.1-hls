"""Freeze or verify the local source package; runtime logs and secrets are excluded."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'runs', 'third_party', '__pycache__', '.git'}


def hashes():
    result = {}
    paths = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE]
        paths.extend(Path(directory)/f for f in files)
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        if (not path.is_file() or any(x in EXCLUDE for x in rel.parts)
                or path.name in ('SOURCE_MANIFEST.json', 'model.local.json', '.env', 'write-check.txt')):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024*1024), b''):
                digest.update(chunk)
        result[rel.as_posix()] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT/'SOURCE_MANIFEST.json'
    actual = hashes()
    if args.check:
        expected = json.loads(path.read_text(encoding='utf-8'))
        changed = sorted(k for k in set(expected)|set(actual) if expected.get(k) != actual.get(k))
        print(json.dumps({'ok': not changed, 'changed': changed, 'file_count': len(actual)}, indent=2))
        return int(bool(changed))
    path.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Frozen files:', len(actual))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
