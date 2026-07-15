from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


TASK_ORDER = [
    "python_palindrome", "python_merge_intervals", "arithmetic_components",
    "reasoning_request_rate", "structured_json", "translation_french",
    "reproducibility_bullets",
]


def response_text(raw: str) -> str:
    text = raw.split("<channel|>")[-1] if "<channel|>" in raw else raw
    text = text.replace("<|channel>thought", "").replace("<turn|>", "")
    return text.strip()


def code_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def validate(task: str, text: str) -> tuple[bool, str]:
    if task == "python_palindrome":
        code = code_block(text)
        tests = "\nassert is_palindrome('Racecar') is True\nassert is_palindrome('A man, a plan, a canal: Panama!') is True\nassert is_palindrome('OpenAI') is False\nassert is_palindrome('') is True\nassert is_palindrome('!!!') is True\n"
        result = subprocess.run([sys.executable, "-I", "-c", code + tests], capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stderr.strip()
    if task == "python_merge_intervals":
        code = code_block(text)
        tests = "\nassert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[5,7]]) == [[5,7]]\n"
        result = subprocess.run([sys.executable, "-I", "-c", code + tests], capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stderr.strip()
    if task == "arithmetic_components":
        return text == "1764", f"answer={text}"
    if task == "reasoning_request_rate":
        return text == "120", f"answer={text}"
    if task == "structured_json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            return False, str(error)
        expected = {"project": "ExpertFlow", "status": "experimental", "risks": ["observer overhead", "runtime complexity"]}
        return value == expected, f"value={value!r}"
    if task == "translation_french":
        lowered = text.casefold()
        checks = {
            "server": "serveur" in lowered,
            "restarted": "redémarr" in lowered,
            "after_update": "après" in lowered and "mise à jour" in lowered,
            "contrast": "mais" in lowered,
            "no_data_lost": "aucune donnée" in lowered and "perdue" in lowered,
        }
        return all(checks.values()), json.dumps(checks, sort_keys=True, ensure_ascii=False)
    if task == "reproducibility_bullets":
        bullets = [line.strip()[1:].strip() for line in text.splitlines() if line.strip().startswith(("*", "-"))]
        counts = [len(re.findall(r"[\w’'-]+", bullet)) for bullet in bullets]
        relevant = all(any(word in bullet.casefold() for word in ("valid", "trust", "verify", "verification", "build")) for bullet in bullets)
        return len(bullets) == 3 and all(count <= 6 for count in counts) and relevant, f"counts={counts}; bullets={bullets!r}"
    raise ValueError(task)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, object] = {"schema_version": "1.0.0", "measurement_kind": "measured", "tasks": {}}
    for task in TASK_ORDER:
        task_result: dict[str, object] = {}
        for mode in ("normal", "observer"):
            run = args.root / "runs" / task / mode
            tokens = json.loads((run / "tokens.json").read_text(encoding="utf-8"))
            measurement = json.loads((run / "measurement.json").read_text(encoding="utf-8"))
            text = response_text(tokens["generated_text"])
            passed, detail = validate(task, text)
            entry: dict[str, object] = {
                "completed": measurement["status"] == "passed",
                "task_passed": passed,
                "detail": detail,
                "generated_token_count": len(tokens["generated_token_ids"]),
                "generated_text": text,
                "duration_seconds_diagnostic": measurement["duration_seconds"],
                "process_gpu_peak_mib": measurement["memory"]["process_gpu_peak_mib"],
                "gpu_before": measurement["memory"]["gpu_before"],
                "gpu_after_settled": measurement["memory"]["gpu_after_settled"],
            }
            if mode == "observer":
                entry["routing_event_count"] = sum(1 for _ in (run / "trace.jsonl").open(encoding="utf-8"))
            task_result[mode] = entry
        result["tasks"][task] = task_result
    normal_passes = sum(bool(result["tasks"][task]["normal"]["task_passed"]) for task in TASK_ORDER)
    observer_passes = sum(bool(result["tasks"][task]["observer"]["task_passed"]) for task in TASK_ORDER)
    retained = sum(bool(result["tasks"][task]["normal"]["task_passed"] and result["tasks"][task]["observer"]["task_passed"]) for task in TASK_ORDER)
    result["aggregate"] = {"normal_passes": normal_passes, "observer_passes": observer_passes, "successful_outcomes_retained": retained}
    result["decision"] = "accept" if observer_passes == 7 and retained >= 6 else "reject"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))
    print(f"decision={result['decision']}")
    return 0 if result["decision"] == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
