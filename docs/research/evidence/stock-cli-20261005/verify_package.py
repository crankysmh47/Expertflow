"""Smoke-test the built wheel from outside the research checkout."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT/'.superpowers/sdd/2026-10-05-stock-cli-and-coverage/wheel-env/Scripts/python.exe'
CLI = PYTHON.with_name('expertflow.exe')
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true',help='archive the final smoke run separately')
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix='expertflow-stock-package-') as cwd:
        def run(args, expected):
            result = subprocess.run(args,cwd=cwd,capture_output=True,text=True)
            if result.returncode != expected:
                raise AssertionError(result.stdout+'\n'+result.stderr)
            checks.append({'command':list(map(str,args)),'exit_code':result.returncode,
                'stdout':result.stdout,'stderr':result.stderr})
            return result.stdout
        location = run([str(PYTHON),'-c','import expertflow.stock.cli; print(expertflow.stock.cli.__file__)'],0)
        assert 'site-packages' in location and str(ROOT/'src') not in location
        assert ',stock,' in run([str(CLI),'--help'],0).replace(' ','')
        assert 'Closed studies' in run([str(CLI),'stock','--help'],0)
        missing = json.loads(run([str(CLI),'stock','search','generate'],3))
        assert missing['status'] == 'ENVIRONMENT-BLOCKED' and 'checkout' in missing['reason']
        assert '--space-config' in run([str(CLI),'stock','--project',str(ROOT),'search','generate','--help'],0)
        coverage = json.loads(run([str(CLI),'stock','--project',str(ROOT),'coverage','inspect'],0))
        assert coverage['status'] == 'REGISTERED-NOT-RUN' and coverage['native_calls'] == 0
    destination = HERE/('packaging-final.json' if args.final else 'packaging.json')
    if destination.exists():
        raise ValueError('packaging record exists; no overwrite')
    wheel = ROOT/'.superpowers/sdd/2026-10-05-stock-cli-and-coverage/dist/expertflow_local-0.1.0-py3-none-any.whl'
    destination.write_text(json.dumps({'status':'PASS','checks':checks,'wheel_sha256':sha256(wheel.read_bytes()).hexdigest(),
        'scope':'Wheel installed without dependencies in a separate environment; help, prerequisites and metadata inspection only.'},
        indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','packaging_checks':len(checks),'native_calls':0}))


if __name__ == '__main__':
    main()
