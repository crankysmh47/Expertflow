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
    assert "object-fit:contain" in html.replace(" ", "")
    assert "scroll-snap-type:ymandatory" in html.replace(" ", "")
    assert "http://" not in html and "https://" not in html
