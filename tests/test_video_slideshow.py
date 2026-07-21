from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_video_slideshow_contains_every_visual_and_fit_contract() -> None:
    html = (ROOT / "submission/demo-video-slideshow.html").read_text(encoding="utf-8")
    for asset in (
        "demo-video-assets/title.svg",
        "demo-video-assets/architecture.svg",
        "demo-video-assets/codex-workflow.svg",
        "demo-video-assets/result.svg",
        "../docs/assets/cache-decision.svg",
        "../docs/assets/placement-map.svg",
        "../docs/assets/profile-cards.svg",
        "demo-video-assets/limitations.svg",
        "demo-video-assets/final-summary.svg",
    ):
        assert asset in html
    compact = "".join(html.split())
    assert html.count("<object") == 9
    assert "object-fit:contain" in compact
    assert "scroll-snap-type:ymandatory" in html.replace(" ", "")
    assert "http://" not in html and "https://" not in html


def test_video_slideshow_frames_static_residency_as_compiler_output() -> None:
    html = (ROOT / "submission/demo-video-slideshow.html").read_text(encoding="utf-8").lower()
    assert "placement compiler" in html
    assert "hardware-specific" in html
    assert "the evidence changed the product" in html
    assert "static residency is one compiled policy" in html


def test_video_slideshow_has_replayable_accessible_motion() -> None:
    html = (ROOT / "submission/demo-video-slideshow.html").read_text(encoding="utf-8")
    assert "IntersectionObserver" in html
    assert "is-active" in html
    assert 'id="replay-current"' in html
    assert "prefers-reduced-motion:reduce" in html.replace(" ", "")
    assert "aria-current" in html
    assert "installSvgMotion" in html
    assert "hydrateSvg" in html
    assert "DOMParser" in html
    assert "fetch(object.data)" in html
