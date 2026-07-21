from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_offline_dashboard_contains_required_panels_and_claim_labels() -> None:
    html = (ROOT / "docs/evidence/product-release/dashboard.html").read_text(encoding="utf-8")
    for text in (
        "Hidden CPU expert bottleneck",
        "Stock whole-layer placement",
        "ExpertFlow expert-bank placement",
        "28.13",
        "22.48%",
        "10,966.801 MiB",
        "49/100",
        "50/100",
        "+2.25%",
        "NO CACHE OPPORTUNITY",
        "35.6699",
        "262,144",
        "Measured recorded evidence",
        "Live benchmark",
        "Simulation",
    ):
        assert text in html
    assert "26.35 TPS" not in html
    assert "<script src=" not in html
    assert 'src="http://' not in html and 'src="https://' not in html
    assert "--pcb:#0b3d20" in html.lower()
    assert "--gold:#d6a84a" in html.lower()
    assert "Codex + GPT-5.6 workflow" in html


def test_dashboard_is_a_narrative_hardware_console_with_runnable_proof_paths() -> None:
    html = (ROOT / "docs/evidence/product-release/dashboard.html").read_text(encoding="utf-8")
    for text in (
        'class="circuit-board"',
        'id="story"',
        'id="proof-actions"',
        'id="placement"',
        'id="evidence-drawer"',
        "PLACEMENT,",
        "COMPILED.",
        "Run the live matched test",
        ".\\scripts\\live-tps-demo.ps1 -Mode Demo",
        "https://github.com/crankysmh47/Expertflow",
        "Replay → Live → Rebuild",
        "prefers-reduced-motion: reduce",
    ):
        assert text in html
    assert html.index("28.13") < html.index("Quality evidence")


def test_dashboard_restores_deep_link_after_reveal_setup() -> None:
    html = (ROOT / "docs/evidence/product-release/dashboard.html").read_text(encoding="utf-8")
    assert "location.hash" in html
    assert "scrollIntoView" in html


def test_submission_claims_ledger_classifies_every_claim() -> None:
    ledger = (ROOT / "submission/claims-ledger.md").read_text(encoding="utf-8")
    assert "| Claim | Class |" in ledger
    for label in ("Measured", "Simulated", "Projected", "Planned"):
        assert f"| {label} |" in ledger or f"| {label} " in ledger
    assert "28.13 TPS" in ledger
    assert "Strict 1% perplexity confidence gate was not met" in ledger
    assert "NO CACHE OPPORTUNITY" in ledger


def test_judge_docs_and_readme_expose_all_product_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "submission/judge-test-guide.md").read_text(encoding="utf-8")
    for command in ("doctor", "profile", "optimize", "run", "serve", "compare", "demo --replay"):
        assert f"expertflow {command}" in readme or f"expertflow {command}" in guide
    assert "A placement compiler for quantized MoE models." in readme
    assert "GGUF is not included" in guide
