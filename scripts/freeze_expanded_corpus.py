from __future__ import annotations

import argparse
import json
from pathlib import Path

from expertflow.expanded_collection import build_frozen_manifest


PROMPTS = {
    "general_instruction": [
        ("morning-routine", "User: Design a calm 25-minute morning routine for a student who must leave by 7:30. Include a two-minute fallback."),
        ("meeting-agenda", "User: Turn these goals into a 30-minute meeting agenda: choose a venue, assign outreach, and confirm accessibility."),
        ("apology-note", "User: Draft a sincere four-sentence apology for returning a borrowed camera late without making excuses."),
        ("decision-checklist", "User: Make a checklist for choosing between repairing an old bicycle and buying a reliable used one."),
        ("concept-explanation", "User: Explain why metal feels colder than wood at the same room temperature using a simple analogy."),
        ("meal-planning", "User: Plan three inexpensive vegetarian lunches that reuse ingredients but do not feel identical."),
        ("study-plan", "User: Build a five-day revision plan for two exams, with retrieval practice and one recovery evening."),
        ("boundary-message", "User: Write a friendly message declining recurring late-night calls while offering two daytime alternatives."),
        ("travel-packing", "User: Create a minimal packing list for a rainy two-day train trip with one formal dinner."),
        ("feedback-rewrite", "User: Rewrite this feedback constructively: Your reports are confusing and always arrive too late."),
        ("event-plan", "User: Propose a low-noise indoor activity plan for eight people with mixed mobility needs."),
        ("comparison-brief", "User: Compare paper and digital note-taking for a field course; give tradeoffs and a recommendation framework."),
        ("care-instructions", "User: Give a safe maintenance checklist for a wooden cutting board, including what cleaners to avoid."),
        ("negotiation-script", "User: Draft a concise script for asking a landlord to repair a leaking window before the next storm."),
    ],
    "code": [
        ("python-parser", "User: Implement a Python parser for key=value lines that ignores comments and reports malformed line numbers. Include tests."),
        ("typescript-retry", "User: Write a typed TypeScript retry helper with capped exponential backoff and injected sleep for deterministic tests."),
        ("sql-dedup", "User: Write PostgreSQL SQL that keeps the newest row per email and explain deterministic tie handling."),
        ("rust-iterator", "User: Implement a Rust iterator that yields fixed-size slices and handles a final short slice without copying."),
        ("go-worker-pool", "User: Sketch a Go worker pool that stops on context cancellation and never leaks result-channel senders."),
        ("javascript-bug", "User: Debug a JavaScript closure bug where buttons in a loop all print the final index; show two fixes."),
        ("cache-design", "User: Design an LRU cache API with O(1) get and put, then state invariants suitable for property tests."),
        ("shell-validation", "User: Write defensive PowerShell pseudocode that validates an input directory before enumerating JSON files."),
        ("regex-review", "User: Review a regex-based email validator, explain its limits, and propose a pragmatic application-level contract."),
        ("api-pagination", "User: Implement cursor pagination pseudocode that avoids duplicates when rows are inserted between requests."),
        ("cpp-lifetime", "User: Explain a dangling string_view bug in C++ and provide a corrected ownership-safe example."),
        ("database-transaction", "User: Show a transaction-safe inventory decrement that prevents overselling under concurrent requests."),
        ("test-refactor", "User: Refactor a brittle test that sleeps for one second into a deterministic event-driven test."),
        ("memory-profile", "User: Diagnose a service whose memory rises when request futures time out; list instrumentation and likely ownership bugs."),
    ],
    "math_reasoning": [
        ("mixture", "User: A 20% solution and a 50% solution make 12 liters at 35%. Find each amount and verify the weighted average."),
        ("combinatorics", "User: Count five-character codes with distinct digits where the first digit is nonzero. Explain each factor."),
        ("geometry", "User: A right triangle has legs differing by 7 and area 60. Find the legs and reject impossible roots."),
        ("probability", "User: Two fair dice are rolled until a sum of 7 appears. Find the expected roll count and justify the distribution."),
        ("recurrence", "User: Solve T(n)=3T(n/3)+n for powers of three using a recursion tree."),
        ("number-proof", "User: Prove that the square of every odd integer is congruent to 1 modulo 8."),
        ("optimization", "User: A rectangle under y=12-x^2 has its base on the x-axis. Derive the maximum area."),
        ("rate-problem", "User: A cyclist travels uphill at 12 km/h and returns at 20 km/h. Find the round-trip average speed."),
        ("logic", "User: Determine whether (P implies Q) and (Q implies R) entail (not R implies not P), with a proof."),
        ("statistics", "User: Compare mean and median for incomes 24, 26, 27, 29, and 140; explain which summary is robust."),
        ("sequence", "User: Derive a closed form for the arithmetic sequence with a3=11 and a9=35, then compute a20."),
        ("bayes", "User: A test is 95% sensitive and 90% specific for a condition with 2% prevalence. Compute the positive predictive value."),
        ("graph-reasoning", "User: Prove that a tree with n vertices has n-1 edges using induction on a leaf."),
        ("constraint-puzzle", "User: Seat Ana, Bo, Cy, and Dev in a row if Ana cannot sit beside Bo and Cy must precede Dev. Count arrangements."),
    ],
    "translation_multilingual": [
        ("urdu-notice", "User: Translate into natural Urdu: The library will close early on Friday for electrical maintenance. Add a literal back-translation."),
        ("spanish-register", "User: Translate Please send the revised invoice by noon into formal and conversational Spanish; explain the register change."),
        ("french-instructions", "User: Translate Keep this medicine out of children's reach into French, then explain the imperative construction without medical advice."),
        ("german-ambiguity", "User: Translate the German sentence Ich habe sie gestern gesehen into English and list its pronoun ambiguity."),
        ("bilingual-summary", "User: Explain cache eviction in simple English, then give a two-sentence summary in Roman Urdu."),
        ("arabic-tone", "User: Translate We appreciate your patience while the system is restored into polite Modern Standard Arabic."),
        ("japanese-politeness", "User: Give polite and casual Japanese versions of Please wait here for five minutes, with romanization."),
        ("portuguese-localization", "User: Localize Tap Save to keep your changes for Brazilian Portuguese; avoid a word-for-word UI translation."),
        ("hindi-code-switch", "User: Hindi mein explain karo why backups should be tested, but keep snapshot and restore as English technical terms."),
        ("italian-backtranslation", "User: Translate The meeting was postponed because the train was delayed into Italian and back-translate literally."),
        ("korean-honorific", "User: Translate Could you check the attachment? into business-polite Korean and explain the honorific choice."),
        ("turkish-contrast", "User: Translate Although the room was small, it felt comfortable into Turkish and identify the contrast marker."),
        ("punjabi-script", "User: Translate Please bring your identification card tomorrow into Punjabi in Shahmukhi script, followed by Roman transliteration."),
        ("mixed-support", "User: Responde en español: explain why a cache miss raises latency, keeping cache miss and latency in English."),
    ],
    "structured_output": [
        ("incident-json", "User: Return only JSON with keys severity, evidence, and next_actions for: login failures rose after a certificate rotation."),
        ("contact-extraction", "User: Extract name, role, email, and deadline as JSON from: Mira Chen, release lead, mira@example.org, needs approval by 18 July."),
        ("csv-normalization", "User: Return CSV with columns item,quantity,unit from: three boxes of clips; 1.5 kilograms of rice; twelve pens."),
        ("yaml-plan", "User: Return valid YAML with owner, tasks, and risks for a two-day documentation sprint led by Sam."),
        ("markdown-table", "User: Produce only a Markdown table comparing PNG, JPEG, and WebP by transparency, typical use, and tradeoff."),
        ("schema-validation", "User: Return JSON Schema for an object with required id, optional tags, and a status enum of queued or complete."),
        ("timeline-extraction", "User: Extract an ordered JSON timeline: deploy 09:00, errors 09:06, rollback 09:14, recovery 09:19."),
        ("classification-jsonl", "User: Return JSONL classifying apple, carrot, and salmon by food_group and plant_based boolean."),
        ("xml-catalog", "User: Return well-formed XML for two books with title and year, using 2021 for Drift and 2024 for Ember."),
        ("sql-output", "User: Return only a parameterized SQL query selecting active orders created after a supplied timestamp."),
        ("risk-matrix", "User: Return JSON array entries with risk, likelihood, impact, mitigation for power loss and disk failure."),
        ("entity-relations", "User: Extract entities and directed relations as JSON from: Noor reports to Ilya, who manages Project Cedar."),
        ("calendar-object", "User: Return only JSON for an event titled Design Review on 2026-07-20 from 14:00 to 14:45 UTC."),
        ("nested-inventory", "User: Convert warehouse notes into JSON grouped by aisle: A has 4 lamps; B has 2 chairs and 7 cushions."),
    ],
    "topic_shift": [
        ("garden-to-code", "User: Recommend two shade plants. Then switch topics and fix a Python off-by-one loop over a list."),
        ("history-to-math", "User: Give one fact about the printing press. New topic: solve 5x-9=31 and check it."),
        ("cooking-to-translation", "User: Suggest a use for leftover rice. Now translate The window is open into French."),
        ("travel-to-json", "User: List two train-trip essentials. Then return JSON with destination Lahore and nights 2."),
        ("physics-to-email", "User: Explain static electricity briefly. Change topic and draft a subject line for a delayed shipment."),
        ("music-to-sql", "User: Define musical tempo. Next, write SQL counting orders by status."),
        ("budget-to-debug", "User: Give one tip for tracking cash expenses. Now diagnose a null dereference in pseudocode."),
        ("language-to-logic", "User: Translate hello into Turkish. Unrelated: determine whether all squares are rectangles."),
        ("weather-to-yaml", "User: Explain why fog forms. Then output YAML with city: Quetta and alert: low_visibility."),
        ("books-to-probability", "User: Recommend a way to organize a bookshelf. Switch topics: find the probability of heads on one fair coin toss."),
        ("exercise-to-rust", "User: Suggest a gentle desk stretch. New task: explain Rust borrowing in two sentences."),
        ("astronomy-to-csv", "User: State why the Moon has phases. Then return CSV for Mars,2 and Venus,1."),
        ("repair-to-spanish", "User: Give a cautious first step for a squeaky door hinge. Now translate Good evening into Spanish."),
        ("geography-to-algorithm", "User: Name the ocean east of Africa. Abruptly switch and describe binary search invariants."),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_frozen_manifest(PROMPTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
