"""Run the compiler's early artifact gate without loading a model or using CUDA."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from expertflow.compiler.preflight import DEFAULT_CUDA_RUNTIME, run_preflight


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stock-dir', type=Path, required=True)
    parser.add_argument('--fork-dir', type=Path, required=True)
    parser.add_argument('--cuda-runtime', type=Path, default=DEFAULT_CUDA_RUNTIME)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(args.root.resolve(), args.model, args.stock_dir, args.fork_dir,
                           cuda_runtime=args.cuda_runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=args.output.parent, prefix=args.output.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary_path, args.output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(json.dumps({'status': result['status'], 'report': str(args.output)}))
    return {'READY': 0, 'ENVIRONMENT-BLOCKED': 3, 'IDENTITY-STOP': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
