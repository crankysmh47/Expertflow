"""Check links/fences in this phase's documentation, without changing files."""
from pathlib import Path
import re
import subprocess


def main():
    paths = set(subprocess.check_output(['git','diff','--name-only'],text=True).splitlines())
    paths.update(str(p) for p in Path('docs/evidence/stock-cli-20261005').glob('*.md'))
    paths.update(('docs/superpowers/plans/2026-10-05-stock-cli-and-coverage.md',
                  'docs/superpowers/specs/2026-10-05-stock-coverage.md'))
    links = 0
    for name in sorted(paths):
        path = Path(name)
        if path.suffix != '.md' or path.name == 'PROJECT_LOG.md':
            continue
        text = path.read_text(encoding='utf-8')
        if len(re.findall(r'^\s*```',text,re.M)) % 2:
            raise ValueError('unbalanced fences: '+name)
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target = match[1].strip().strip('<>')
            if re.match(r'^[a-z]+:',target,re.I) or target.startswith('#'):
                continue
            if not (path.parent/target.split('#')[0]).resolve().exists():
                raise ValueError(f'{name}: missing {target}')
            links += 1
    print(f'PASS: {links} local Markdown links and balanced fences')


if __name__ == '__main__':
    main()
