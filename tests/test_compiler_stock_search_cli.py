import importlib.util
import json
from pathlib import Path

from expertflow.compiler.evidence import EvidenceStore


def driver():
    spec = importlib.util.spec_from_file_location('stock_search_driver',
        Path('scripts/benchmark_compiler_stock_search.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_prerequisite_stops_before_loading_weights(tmp_path,monkeypatch,capsys):
    module = driver()
    monkeypatch.setattr(module,'load_compiler_inputs',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('no weights load')))
    code = module.main(['--action','run','--source-evidence-db',str(tmp_path/'missing'),
        '--evidence-db',str(tmp_path/'target.sqlite3'),'--output-dir',str(tmp_path/'run')])
    assert code == 3 and json.loads(capsys.readouterr().out)['status'] == 'ENVIRONMENT-BLOCKED'
    assert not (tmp_path/'target.sqlite3').exists()


def test_existing_target_is_rejected_before_loading_weights(tmp_path,monkeypatch,capsys):
    module = driver()
    source = EvidenceStore(tmp_path/'source.sqlite3')
    target = EvidenceStore(tmp_path/'target.sqlite3')
    monkeypatch.setattr(module,'load_compiler_inputs',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('no weights load')))
    code = module.main(['--action','run','--source-evidence-db',str(source.path),
        '--evidence-db',str(target.path),'--output-dir',str(tmp_path/'run')])
    assert code == 2 and json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'


def test_generate_writes_preview_without_instantiating_native_runner(tmp_path,monkeypatch,capsys):
    module = driver()
    source = EvidenceStore(tmp_path/'source.sqlite3')
    monkeypatch.setattr(module,'load_compiler_inputs',lambda *args,**kwargs: object())
    monkeypatch.setattr(module,'capture_host_environment',lambda:{'fixture':True})
    monkeypatch.setattr(module,'prepare_search',lambda *args,**kwargs:{'manifest_sha256':'a'*64})
    def forbidden(*args,**kwargs):
        raise AssertionError('generate must not instantiate native runner/sampler')
    monkeypatch.setattr(module,'WindowsGpuMemorySampler',forbidden)
    monkeypatch.setattr(module,'ServerMeasurementRunner',forbidden)
    preview = tmp_path/'preview.json'
    code = module.main(['--action','generate','--source-evidence-db',str(source.path),
        '--output-dir',str(tmp_path/'future-run'),'--manifest-output',str(preview)])
    assert code == 0 and json.loads(capsys.readouterr().out)['status'] == 'GENERATED-SEARCH-MANIFEST'
    assert json.loads(preview.read_text())['manifest_sha256'] == 'a'*64
    assert not (tmp_path/'future-run').exists()


def test_accepted_execute_uses_existing_store_without_collection_source_database(tmp_path,monkeypatch,capsys):
    module = driver()
    database = EvidenceStore(tmp_path/'search.sqlite3').path
    inp = object()
    monkeypatch.setattr(module,'load_compiler_inputs',lambda *args,**kwargs:inp)
    def execute(directory,store,actual,output):
        assert directory == tmp_path/'recommended' and store.path == database
        assert actual is inp and output == tmp_path/'execution'
        return 'MEASURED-ACCEPTED-STOCK-SEARCH',{'measurement_id':'fixture-only'}
    monkeypatch.setattr(module,'run_search_recommendation',execute)
    code = module.main(['--action','execute','--source-evidence-db',str(tmp_path/'missing'),
        '--evidence-db',str(database),'--recommendation',str(tmp_path/'recommended'),
        '--output-dir',str(tmp_path/'execution')])
    assert code == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'MEASURED-ACCEPTED-STOCK-SEARCH'


def test_accepted_execute_rejects_existing_output_before_loading_inputs(tmp_path,monkeypatch,capsys):
    module = driver()
    database = EvidenceStore(tmp_path/'search.sqlite3').path
    output = tmp_path/'execution';output.mkdir()
    def forbidden(*args,**kwargs):
        raise AssertionError('existing output must fail before input loading or native execution')
    monkeypatch.setattr(module,'load_compiler_inputs',forbidden)
    code = module.main(['--action','execute','--evidence-db',str(database),
        '--recommendation',str(tmp_path/'recommended'),'--output-dir',str(output)])
    assert code == 2 and json.loads(capsys.readouterr().out)['status'] == 'IDENTITY-STOP'
