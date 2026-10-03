from source_contract_paths import LLAMA


def test_phase_markers_are_explicit_and_default_off():
    server=(LLAMA/'tools/server/server-context.cpp').read_text(encoding='utf-8')
    common=(LLAMA/'common/common.cpp').read_text(encoding='utf-8')
    backend=(LLAMA/'ggml/src/ggml-backend.cpp').read_text(encoding='utf-8')
    context=(LLAMA/'src/llama-context.cpp').read_text(encoding='utf-8')
    assert 'llama_set_expertflow_profile_phase(ctx_tgt' in server
    assert 'SLOT_STATE_GENERATING ? 3 : 2' in server
    assert 'llama_set_expertflow_profile_phase(lctx, 1)' in common
    assert 'ggml_backend_sched_profile_note_tokens(sched.get(), ubatch.n_tokens)' in context
    assert 'graph->nodes[graph->n_nodes - 1]->ne[1] == 1' not in backend
    assert 'expertflow_split_profile_records[5][EXPERTFLOW_SPLIT_PROFILE_MAX_RECORDS]' in backend
    assert 'if (!sched->expertflow_split_profile_enabled)' in backend


def test_server_phase_only_considers_batch_token_owners():
    server=(LLAMA/'tools/server/server-context.cpp').read_text(encoding='utf-8')
    assert 'batch_view.seq_id[token][seq] == slot.id' in server
    assert 'batch_view.n_seq_id[token]' in server
