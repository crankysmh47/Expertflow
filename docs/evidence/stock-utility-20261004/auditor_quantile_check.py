import math
from pathlib import Path
import random
import runpy

module=runpy.run_path('docs/evidence/stock-utility-20261004/independent_audit.py')
direct=[20.0]*10
selected=[20.0+.3*n for n in range(10)]
logs=[math.log(b)-math.log(a) for a,b in zip(direct,selected)]
rng=random.Random(20261003)
draws=sorted(100*math.expm1(sum(rng.choices(logs,k=10))/10) for _ in range(10000))
actual=module['paired'](direct,selected)
assert actual['ci90_pct']==[draws[499],draws[9499]], 'registered empirical percentile uses nearest rank, not interpolation'
assert actual['ci95_pct']==[draws[249],draws[9749]]
print('registered nearest-rank percentile known-vector PASS')
