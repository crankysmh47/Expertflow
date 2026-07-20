import hashlib
import json
from pathlib import Path

from expertflow.product.commands import sha256_canonical_json


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "docs/evidence/product-release"


def test_frozen_release_state_reconstructs_verified_q6_result() -> None:
    state = json.loads((EVIDENCE / "release-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "RELEASE STATE VALID"
    assert state["source"]["expertflow_commit"] == "b24eb1f5ed21e3ef5ecc4a53a45af97ff34460eb"
    assert state["source"]["product_result_commit"] == "db3b5c5ea2857cbdc4bff319cb7e47fa410c889b"
    assert state["source"]["llama_commit"] == "451224ab4d12a616dc3e16e8c8063f4b331f531c"
    assert state["model"]["sha256"] == "089ecf3bbad0b18b187ff1b3de171413f8a5d8fb246bc1b776a68c95ad9a07ba"
    assert state["runtime"]["llama_cli_sha256"] == "5d68046dcd26e2fd018aaeaad5f99cdb7d88eca6fc10935925f1d660f7009407"
    assert state["placement"]["layers"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20]
    assert state["placement"]["experts_per_layer"] == 128
    assert state["performance"] == {
        "expertflow_decode_tps": 28.13,
        "strongest_stock_decode_tps": 22.967,
        "improvement_pct": 22.480080114947533,
        "peak_process_owned_mib": 10966.801,
    }
    assert state["quality"]["mmlu_off"] == 49
    assert state["quality"]["mmlu_on"] == 50
    assert state["quality"]["ppl_95_upper_pct"] == 2.2529372655983686
    assert state["quality"]["strict_ppl_gate_pass"] is False
    assert state["cache_strategy"]["verdict"] == "NO CACHE OPPORTUNITY"


def test_release_state_evidence_hashes_match_committed_files() -> None:
    state = json.loads((EVIDENCE / "release-state.json").read_text(encoding="utf-8"))
    for item in state["evidence"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256_canonical_json(path) == item["sha256"]


def test_canonical_json_hash_ignores_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    payload = '{\n  "value": 1\n}\n'
    lf.write_bytes(payload.encode("utf-8"))
    crlf.write_bytes(payload.replace("\n", "\r\n").encode("utf-8"))
    assert sha256_canonical_json(lf) == sha256_canonical_json(crlf)


def test_immutable_deployment_has_no_private_absolute_paths() -> None:
    deployment = json.loads((EVIDENCE / "deployment-result.json").read_text(encoding="utf-8"))
    serialized = json.dumps(deployment)
    assert "Hank47" not in serialized
    assert "C:\\models" not in serialized
    assert deployment["model"]["path_env"] == "EXPERTFLOW_MODEL_PATH"
    assert deployment["runtime"]["binary_env"] == "EXPERTFLOW_LLAMA_CLI"
    assert deployment["environment"]["LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER"] == "0,1,2,3,4,5,6,7,8,9,15,20"
    assert deployment["evidence"]["decode_tps"] == 28.13
