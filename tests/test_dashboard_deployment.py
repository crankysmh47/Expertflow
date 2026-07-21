import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_vercel_routes_root_and_dashboard_to_the_static_observatory() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    rewrites = {(item["source"], item["destination"]) for item in config["rewrites"]}
    target = "/docs/evidence/product-release/dashboard.html"
    assert ("/", target) in rewrites
    assert ("/dashboard", target) in rewrites
    assert config["cleanUrls"] is True


def test_deployment_guide_covers_public_replay_and_live_hardware_paths() -> None:
    guide = (ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")
    for text in (
        "https://github.com/crankysmh47/Expertflow",
        "npx vercel --prod",
        "uv run expertflow demo --replay",
        ".\\scripts\\live-tps-demo.ps1 -Mode Demo",
        "089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba",
        "Windows 11 x64",
        "Linux x64 + NVIDIA",
        "The GGUF is not included",
    ):
        assert text in guide
