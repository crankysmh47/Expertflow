from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_canonical_product_guide_explains_architecture_and_evidence_boundaries() -> None:
    guide_path = ROOT / "docs/PRODUCT.md"
    assert guide_path.exists()
    guide = guide_path.read_text(encoding="utf-8")
    for text in (
        "# ExpertFlow Product and Architecture Guide",
        "Profile -> Compile -> Place -> Run -> Verify",
        "Stock execution boundary",
        "Complete expert-bank bundle",
        "gate/up",
        "down",
        "Identity remapping",
        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20",
        "hardware-specific emitted plan",
        "28.13",
        "22.967",
        "10,966.801 MiB",
        "No eviction",
        "GPT-5.6",
        "Codex",
        "docs/evidence/product-release/release-scorecard.json",
    ):
        assert text in guide
    for asset in (
        "assets/architecture.svg",
        "assets/placement-map.svg",
        "assets/cache-decision.svg",
        "assets/profile-cards.svg",
    ):
        assert asset in guide


def test_product_guide_is_linked_and_packaged() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dashboard = (ROOT / "docs/evidence/product-release/dashboard.html").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build_product_release.py").read_text(encoding="utf-8")
    assert "docs/PRODUCT.md" in readme
    assert 'href="https://github.com/crankysmh47/Expertflow/blob/main/docs/PRODUCT.md"' in dashboard
    assert '("docs/PRODUCT.md", "docs/PRODUCT.md")' in builder


def test_readme_uses_dashboard_gallery_instead_of_retired_visual_cards() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/dashboard-architecture.png" in readme
    assert (ROOT / "docs/assets/dashboard-architecture.png").exists()
    for retired in (
        "architecture.svg", "result.svg", "placement-map.svg", "cache-decision.svg",
        "profile-cards.svg", "submission/demo-video-assets/codex-workflow.svg",
    ):
        assert retired not in readme
    for label in ("Repository", "Product architecture", "Live dashboard", "Deployment guide"):
        assert label in readme
