from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding='utf-8')
assert 'call        ggml_compute_forward_mul_mat_id_one_chunk' in text, 'missing compiled expert caller'
assert re.search(r'\bprefetcht0\b', text), 'Q6 expert caller object has no T0 cache hint'
print('compiled expert caller contains T0 hint')
