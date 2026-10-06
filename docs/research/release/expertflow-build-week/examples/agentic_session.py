from concurrent.futures import ThreadPoolExecutor

from openai_client import chat


def main() -> None:
    print("Interactive coding request:\n", chat("Review a Python SHA-256 verifier for likely edge cases."))
    with ThreadPoolExecutor(max_workers=2) as workers:
        replies = list(workers.map(chat, ["Draft unit tests for a JSON manifest loader.", "Draft a PowerShell health check for a local server."]))
    print("Parallel workers:\n", "\n---\n".join(replies))
    repository_context = "A repository contains a CLI, deployment JSON, tests, and an offline report. " * 80
    print("Repository analysis:\n", chat(repository_context + "Identify the three highest-risk release gaps."))


if __name__ == "__main__":
    main()
