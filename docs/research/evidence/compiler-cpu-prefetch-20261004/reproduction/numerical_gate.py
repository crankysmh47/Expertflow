import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path('.superpowers/sdd/cpu-expert-prefetch-20261004')
stock = Path('C:/models/expertflow/builds/llama-a7312ae-cuda128-clean')
candidate = Path('C:/models/expertflow/builds/llama-cpu-prefetch-20261004')
source = Path('C:/models/expertflow/worktrees/llama-cpu-prefetch-20261004')
base = Path('C:/models/expertflow/dependencies/llama.cpp-a7312ae-git')
dumpbin = 'C:/BuildTools2022/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/dumpbin.exe'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def dump(mode, file):
    return subprocess.check_output([dumpbin, mode, str(file)], text=True)

def code(text):
    return text[text.index('File Type:'):].split('  Summary')[0]

def compile_entries(build):
    entries = {}
    for section in (build/'build.ninja').read_text().split('\nbuild '):
        if '.obj: ' not in section or 'ggml-cpu.dir' not in section.splitlines()[0]:
            continue
        lines = section.splitlines()
        entries[lines[0].split(': ')[0]] = [line.strip() for line in lines if line.startswith(('  FLAGS =', '  DEFINES ='))]
    return entries

assert compile_entries(stock) == compile_entries(candidate), 'CPU source set/compile flags differ'
changed = subprocess.check_output(['git', '-C', str(source), 'diff', 'a7312ae94f801fc9c6786dc56e38df57b964f697', '--name-only'], text=True).splitlines()
assert changed == ['ggml/src/ggml-cpu/ggml-cpu.c'], changed
cpu_before = (base/'ggml/src/ggml-cpu/ggml-cpu.c').read_text()
cpu_after = (source/'ggml/src/ggml-cpu/ggml-cpu.c').read_text()
hint = '''#if defined(__AVX__) || defined(__AVX2__)
                    if (type == GGML_TYPE_Q6_K && ir0 + 1 < ir0_end) {
                        _mm_prefetch(src0_cur + (ir0 + 1)*nb01, _MM_HINT_T0);
                    }
#endif
'''
assert cpu_after.count(hint) == 1
assert cpu_after.replace(hint, '') == cpu_before, 'arithmetic/caller source changed beyond hint'
objects = []
for relative in ('ggml-cpu/quants.c.obj', 'ggml-cpu/arch/x86/quants.c.obj'):
    obj = Path('ggml/src/CMakeFiles/ggml-cpu.dir')/relative
    a, b = dump('/DISASM', stock/obj), dump('/DISASM', candidate/obj)
    assert code(a) == code(b), 'quantization/dot-product machine code changed: '+relative
    stem = relative.replace('/', '-').replace('.obj', '')
    (root/(stem+'-stock.disasm')).write_text(a)
    (root/(stem+'-candidate.disasm')).write_text(b)
    objects.append({'object': relative, 'normalized_disassembly_sha256': hashlib.sha256(code(a).encode()).hexdigest()})

def exports(path):
    return re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)', dump('/EXPORTS', path), re.M)

def imports(path):
    text = dump('/IMPORTS', path)
    return sorted(re.findall(r'^\s+([\w.-]+\.dll)\s*$', text, re.M)), sorted(re.findall(r'^\s+[0-9A-F]+\s+(\S+)\s*$', text, re.M))

assert exports(stock/'bin/ggml-cpu.dll') == exports(candidate/'bin/ggml-cpu.dll'), 'CPU export ABI changed'
assert imports(stock/'bin/ggml-cpu.dll') == imports(candidate/'bin/ggml-cpu.dll'), 'CPU import ABI changed'
manifest = json.loads(Path('configs/compiler/runtime-stock.json').read_text())
stock_fixture = candidate/'fixture-stock'
stock_fixture.mkdir(exist_ok=True)
for name, digest in {**manifest['binaries'], **manifest['dependencies']}.items():
    assert sha(stock/'bin'/name) == digest, 'stock pin drift: '+name
    shutil.copy2(stock/'bin'/name, stock_fixture/name)
    if name != 'ggml-cpu.dll':
        shutil.copy2(stock/'bin'/name, candidate/'bin'/name)
fixture = stock/'bin/cpu-expert-prefetch-fixture.exe'
if fixture.exists():
    shutil.move(str(fixture), str(stock_fixture/fixture.name))
shutil.copy2(stock_fixture/'cpu-expert-prefetch-fixture.exe', candidate/'bin/cpu-expert-prefetch-fixture.exe')
subprocess.run([str(stock_fixture/'cpu-expert-prefetch-fixture.exe'), str(root/'stock-fixture.bin')], check=True)
subprocess.run([str(candidate/'bin/cpu-expert-prefetch-fixture.exe'), str(root/'candidate-fixture.bin')], check=True)
assert (root/'stock-fixture.bin').read_bytes() == (root/'candidate-fixture.bin').read_bytes(), 'bitwise expert output mismatch'
result = {'status': 'PASS-NUMERICAL-GATE', 'fixtures': 48, 'threads': [1,12], 'rows': [1,15,16,17,33,257],
          'tokens': [1,3], 'widths': [256,768], 'weight_type': 'Q6_K', 'activation_conversion': 'unchanged F32 to Q8_K',
          'output_bytes': (root/'stock-fixture.bin').stat().st_size,
          'output_sha256': sha(root/'stock-fixture.bin'), 'bitwise_outputs_equal': True,
          'compile_flags_and_source_set_equal': True, 'cpu_exports_and_imports_equal': True,
          'only_source_change': changed[0], 'quantization_dot_disassembly': objects,
          'cpu_compile_commands': compile_entries(candidate)}
(root/'numerical-gate.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'cpu_compile_commands'}))
