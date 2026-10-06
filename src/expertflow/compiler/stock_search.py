"""Bounded scheduling spaces; generation alone confers no numerical eligibility."""

from dataclasses import dataclass, replace
import math
import random
import statistics

from .plan import CandidatePlan, CandidateStatus
from .schema import canonical_payload, canonical_sha256, require_int, require_number


def topology_anchors(host, incumbent_threads):
    cpus = host.get('cpu')
    if not isinstance(cpus, list) or len(cpus) != 1:
        raise ValueError('scheduling requires an unambiguous single-socket topology')
    physical, logical = cpus[0].get('cores'), cpus[0].get('logical_processors')
    require_int(physical, 'physical cores')
    require_int(logical, 'logical processors')
    require_int(incumbent_threads, 'incumbent threads')
    if not physical <= logical <= 64 or incumbent_threads > logical:
        raise ValueError('unsupported topology or incumbent thread count')
    process, system = host.get('process_affinity_mask'), host.get('system_affinity_mask')
    require_int(process, 'process affinity')
    require_int(system, 'system affinity')
    if process != system or system.bit_count() != logical:
        raise ValueError('partial affinity or processor-group topology is unsupported')
    return tuple(sorted({physical, physical+(logical-physical)//2, logical, incumbent_threads}))


def semantic_fingerprint(candidate):
    """Identity for comparisons that vary exactly threads and graph mode."""
    identities = canonical_payload(candidate.identities)
    workload = identities.pop('workload')
    identities.pop('workload_sha256')
    workload.pop('threads')
    workload.pop('cuda_graphs')
    settings = canonical_payload(candidate.settings)
    settings.pop('cuda_graphs')
    return canonical_sha256({'identities':identities, 'workload':workload, 'settings':settings})


@dataclass(frozen=True, slots=True)
class SchedulingSpace:
    candidates: tuple[CandidatePlan, ...]
    excluded_threads: tuple[tuple[int, str], ...]
    untested_threads: tuple[int, ...]


def scheduling_space(base, host, *, graph_modes=('on','off'), excluded_threads=None):
    if base.settings.static is not None or base.settings.cuda_pdl is not None:
        raise ValueError('scheduling space requires pristine fixed placement without PDL/static controls')
    modes = tuple(graph_modes)
    if not modes or any(mode not in ('on','off') for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError('invalid graph mode coverage')
    if base.settings.cuda_graphs not in modes:
        raise ValueError('incumbent graph mode must remain covered')
    anchors = topology_anchors(host, base.identities.workload.threads)
    exclusions = dict(excluded_threads or {})
    for threads,reason in exclusions.items():
        require_int(threads, 'excluded thread count')
        if threads not in anchors or not isinstance(reason,str) or not reason.strip():
            raise ValueError('exclusion must name a topology anchor and reason')
    if base.identities.workload.threads in exclusions:
        raise ValueError('incumbent cannot be excluded')
    candidates = []
    for threads in anchors:
        if threads in exclusions:
            continue
        for mode in sorted(modes, key=lambda m: ('on','off').index(m)):
            workload = replace(base.identities.workload, threads=threads, cuda_graphs=mode)
            identities = replace(base.identities, workload=workload, workload_sha256=canonical_sha256(workload))
            candidates.append(CandidatePlan(identities, replace(base.settings,cuda_graphs=mode),
                status=CandidateStatus.UNMEASURED))
    tested = {c.identities.workload.threads for c in candidates}
    logical = host['cpu'][0]['logical_processors']
    return SchedulingSpace(tuple(candidates),tuple(sorted(exclusions.items())),
        tuple(i for i in range(1,logical+1) if i not in tested))


def screening_schedule(candidate_ids):
    ids = tuple(candidate_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not isinstance(cid,str) or not cid for cid in ids):
        raise ValueError('screening requires unique candidate identities')
    rng = random.Random(20261004)
    blocks = []
    for _ in range(3):
        order = sorted(ids)
        rng.shuffle(order)
        blocks.append(tuple(order))
    return tuple(blocks)


def rank_screening(rows, incumbent_id, schedule):
    """Rank a complete frozen screen; publication still requires native proof."""
    if len(schedule) != 3 or not schedule[0] or incumbent_id not in schedule[0]:
        raise ValueError('invalid screening schedule/incumbent')
    ids = set(schedule[0])
    if any(len(block) != len(ids) or set(block) != ids for block in schedule):
        raise ValueError('screening schedule has missing/duplicate candidates')
    expected = [(block,cid) for block,order in enumerate(schedule) for cid in order]
    if len(rows) != len(expected):
        raise ValueError('screening incomplete')
    rates = {}
    for row,(block,cid) in zip(rows,expected):
        if type(row.get('block')) is not int or (row['block'],row.get('candidate_id')) != (block,cid):
            raise ValueError('screening rows do not match frozen order')
        rate = row.get('decode_tps')
        require_number(rate,'screening native TPS',1e-12)
        rates[block,cid] = rate
    ranked = []
    for cid in ids:
        ratios = [rates[block,cid]/rates[block,incumbent_id] for block in range(3)]
        ranked.append({'candidate_id':cid,'geometric_ratio':math.exp(statistics.mean(
            math.log(rates[block,cid])-math.log(rates[block,incumbent_id]) for block in range(3))),
            'block_ratios':ratios,'block_ratio_range':[min(ratios),max(ratios)],
            'block_ratio_cv_pct':100*statistics.stdev(ratios)/statistics.mean(ratios),
            'uncertainty_scope':'three-block descriptive variation, not confirmation evidence'})
    return tuple(sorted(ranked,key=lambda row:(-row['geometric_ratio'],
        row['candidate_id'] != incumbent_id,row['candidate_id'])))
