import argparse
import json

from expertflow.analysis.narrow_placement import compare_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_prefix")
    parser.add_argument("candidate_prefix")
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.reference_prefix, args.candidate_prefix), indent=2))


if __name__ == "__main__":
    main()
