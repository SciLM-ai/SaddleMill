import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from saddlemill.sella_diagnostics import SellaPassiveQNRecorder

class H:
    update_method='bfgs'; symm=2; block_size=1; nvec=1; cond=1.0
    def asarray(self): return np.array([[2.0,0.0],[0.0,-1.0]])
class Pes:
    def __init__(self): self.H=H(); self.neval=5; self.x=np.array([1.0,2.0])
class Opt:
    def __init__(self): self.pes=Pes(); self.nsteps=0; self.delta=0.1

def test_sella_passive_recorder_changes_no_optimizer_or_pes_state(tmp_path):
    opt=Opt(); before=(opt.nsteps,opt.delta,opt.pes.neval,opt.pes.x.copy())
    rec=SellaPassiveQNRecorder(opt,tmp_path/'sella.jsonl',attempt_id=1)
    rec(); rec.close()
    after=(opt.nsteps,opt.delta,opt.pes.neval,opt.pes.x.copy())
    assert before[:3]==after[:3]
    np.testing.assert_array_equal(before[3],after[3])
    assert (tmp_path/'sella.jsonl').read_text().count('\n')==1
