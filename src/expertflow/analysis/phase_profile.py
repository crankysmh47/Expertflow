"""Fail-closed analysis of explicitly phase-labeled diagnostic split profiles."""

def analyze_phase_profile(data, *, prompt_tokens, generated_tokens, expert_layers):
    if data.get('schema_version') != '2.0.0' or data.get('diagnostic_synchronization') is not True:
        raise ValueError('requires schema2 synchronized diagnostic profile')
    phases = {}
    for phase in data.get('phases', []):
        name = phase.get('phase')
        if name in phases or name not in {'initialization_or_other','warmup','prefill','decode','mixed'}:
            raise ValueError('invalid or duplicate phase')
        for field in ('tokens','graph_calls','split_count'):
            if type(phase.get(field)) is not int or phase[field] < 0:
                raise ValueError('invalid phase accounting')
        phases[name] = phase
        if phase['split_count'] > 256 or (phase['graph_calls'] == 0 and
                (phase['tokens'] != 0 or phase['split_count'] != 0)):
            raise ValueError('inconsistent empty phase or excessive splits')
    if (phases.get('prefill', {}).get('tokens') != prompt_tokens or
            phases.get('decode', {}).get('tokens') != generated_tokens - 1 or
            phases.get('decode', {}).get('graph_calls') != generated_tokens - 1 or
            phases.get('mixed', {}).get('graph_calls', 0) != 0):
        raise ValueError('phase token/call accounting differs from frozen request')
    seen = set();decode = [];experts = set()
    for record in data.get('records', []):
        phase = record.get('phase');split = record.get('split_id')
        if phase not in phases or phase == 'mixed' or type(split) is not int or not 0 <= split < 256:
            raise ValueError('invalid split phase/id')
        if (phase,split) in seen:
            raise ValueError('duplicate phase/split')
        seen.add((phase,split))
        for field in ('calls','input_boundary_us','compute_submit_us','completion_us','total_us'):
            if type(record.get(field)) is not int or record[field] < 0:
                raise ValueError('invalid split accounting')
        if record['calls'] != phases[phase]['graph_calls'] or record['total_us'] != sum(
                record[field] for field in ('input_boundary_us','compute_submit_us','completion_us')):
            raise ValueError('incomplete split calls or inconsistent timer sum')
        if record.get('backend') not in {'CPU','CUDA0'}:
            raise ValueError('unsupported backend')
        if phase == 'decode':
            decode.append(record)
            name = record.get('first_node','')
            if record['backend'] == 'CPU' and name.startswith('ffn_moe_gate_up-'):
                try: layer = int(name.rsplit('-',1)[1])
                except ValueError as error: raise ValueError('invalid expert layer') from error
                if layer in experts: raise ValueError('duplicate expert layer')
                experts.add(layer)
    for phase, accounting in phases.items():
        if {split for name,split in seen if name==phase} != set(range(accounting['split_count'])):
            raise ValueError('incomplete phase split coverage')
    if not decode or experts != set(expert_layers):
        raise ValueError('missing decode/expert coverage')
    total = sum(r['total_us'] for r in decode)
    if total <= 0: raise ValueError('empty decode duration')
    cpu = sum(r['compute_submit_us'] for r in decode if r['backend']=='CPU')
    return {'diagnostic_only':True,'decode_total_us':total,'cpu_compute_us':cpu,
            'cpu_compute_share_pct':100*cpu/total,
            'input_boundary_us':sum(r['input_boundary_us'] for r in decode),
            'cuda_host_submission_us':sum(r['compute_submit_us'] for r in decode if r['backend']=='CUDA0'),
            'cuda_completion_wait_us':sum(r['completion_us'] for r in decode if r['backend']=='CUDA0'),
            'cpu_completion_wait_us':sum(r['completion_us'] for r in decode if r['backend']=='CPU'),
            'pure_transfer_us':None,'phases':phases,
            'limitation':'Synchronized split timing perturbs overlap; input boundaries combine routing/copies/waits.'}
