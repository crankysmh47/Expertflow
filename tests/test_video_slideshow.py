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
        "demo-video-assets/reproduction.svg",
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


def test_video_slideshow_uses_background_circuit_flow_without_foreground_flashes() -> None:
    html = (ROOT / "submission/demo-video-slideshow.html").read_text(encoding="utf-8")
    assert "circuit-field" in html
    assert "trace-flow" in html
    assert "flow-through" in html
    assert "node-glow" in html
    assert "trace-copper" in html
    assert "via-ring" in html
    assert "pcb-chip" in html
    assert "pcb-grain" in html
    assert "room-scan" not in html
    assert "@keyframes scan" not in html
    assert "#f4f2e918" not in html


def test_video_slideshow_uses_judge_reproduction_instead_of_a_weakness_reel() -> None:
    html = (ROOT / "submission/demo-video-slideshow.html").read_text(encoding="utf-8")
    assert "JUDGE REPRODUCTION" in html
    assert "Replay the evidence. Run the live pair. Rebuild the pinned runtime." in html
    assert "Evidence limitations" not in html


def test_final_video_script_is_personal_and_centers_the_live_matched_run() -> None:
    script = (ROOT / "submission/demo-video-script-final.md").read_text(encoding="utf-8")
    for text in (
        "I've always tried to get the most out of the hardware I own.",
        ".\\scripts\\live-tps-demo.ps1 -Mode Demo",
        "fresh matched stock and ExpertFlow processes",
        "one live rehearsal",
        "authoritative ten-pair result",
        "28.13 TPS",
        "22.967 TPS",
    ):
        assert text in script
    assert "No webcam is needed" in script
