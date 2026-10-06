import json
import pytest


def test_missing_and_tampered_job_receipts_are_rejected(tmp_path):
    from expertflow.product.reports import verify_job
    from expertflow.product.runtime import file_digest
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"kind":"local-benchmark", "maximum_model_processes":1}))
    report = {"status":"MEASURED", "model_processes":1,"registration_sha256":file_digest(registration),
              "measurement":{"status":"MEASURED"}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt|measurement"):
        verify_job(tmp_path)
    registration.write_text('{}')
    with pytest.raises(ValueError, match="registration"):
        verify_job(tmp_path)


def test_support_export_excludes_user_paths_and_prompt_text(tmp_path):
    from expertflow.product.reports import support_summary
    registration={"profile":{"model":{"path":"C:/Users/secret/private.gguf", "sha256":"a"*64},
                               "settings":{"context":4096}},"train_prompt":"secret prompt"}
    (tmp_path / "registration.json").write_text(json.dumps(registration))
    (tmp_path / "report.json").write_text(json.dumps({"status":"INCONCLUSIVE","reason":"C:/Users/secret/private.gguf failed","wall_seconds":2,"model_processes":0}))
    text=json.dumps(support_summary(tmp_path))
    assert "secret" not in text and "Users" not in text and "secret prompt" not in text
    assert "INCONCLUSIVE" in text


def test_verified_label_without_evidence_cannot_launch(tmp_path):
    from expertflow.product.profiles import verify_profile
    with pytest.raises(ValueError, match="evidence"):
        verify_profile({"status":"VERIFIED-IMPROVEMENT","evidence":[],"model":{"files":[]}})
