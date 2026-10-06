"""Stage only this phase's new log entry, preserving unrelated working bytes."""
from pathlib import Path
import hashlib
import subprocess
import sys


def main():
    path = Path('PROJECT_LOG.md')
    original = path.read_bytes()
    marker = sys.argv[1].encode()
    working = original.replace(b'\r\n',b'\n')
    base = subprocess.check_output(['git','show','HEAD:PROJECT_LOG.md'])
    if marker in base or working.count(marker) != 1:
        raise ValueError('requires one new unique phase marker')
    addition = working[working.index(marker):]
    staged = base.rstrip(b'\n')+b'\n\n'+addition.rstrip(b'\n')+b'\n'
    digest = subprocess.run(['git','hash-object','-w','--stdin'],input=staged,
        capture_output=True,check=True).stdout.decode().strip()
    subprocess.run(['git','update-index','--cacheinfo',f'100644,{digest},PROJECT_LOG.md'],check=True)
    if path.read_bytes() != original or subprocess.check_output(['git','show',':PROJECT_LOG.md']) != staged:
        raise ValueError('working bytes or selectively staged entry changed')
    print('PASS: only the new project-log entry staged; working bytes preserved',hashlib.sha256(original).hexdigest())


if __name__ == '__main__':
    main()
