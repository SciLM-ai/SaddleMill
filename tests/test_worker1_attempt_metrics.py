import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# load diagnostics_io under canonical module name first
for name in ('diagnostics_io','attempt_metrics'):
    p=ROOT/'saddlemill'/f'{name}.py'; spec=importlib.util.spec_from_file_location(f'saddlemill.{name}',p); m=importlib.util.module_from_spec(spec); sys.modules[f'saddlemill.{name}']=m; spec.loader.exec_module(m)
metrics=sys.modules['saddlemill.attempt_metrics']

class Ctl:
    def get_counter(self,k): assert k=='forcecalls'; return 17
class Acc:
    physical_total_pes_calls=13
class Hist:
    accounting=Acc()
class DAtoms:
    control=Ctl(); canonical_force_history=Hist()
class Opt:
    nsteps=9; last_step_diagnostics={'projected_fmax':0.04}
class Run:
    saddle_engine='ase'; src_index=4; rank=2; config_dict={'Main':{'fmax':0.05}}
class Ctx:
    run=Run(); attempt_id=3; selected_index=7; configured_reaction_type='random'; initial_reaction_type='random'; status='converged'; stop_reason=None; converged=True; forces=np.array([[0.03,0,0],[0.04,0,0]]); dim_rlx=Opt(); d_atoms=DAtoms(); n_force_calls=17; realized_initial_geometry_sha256='sha256:g'; attempt_start_perf_ns=0; atoms=None

def test_exact_physical_counter_is_not_legacy_counter(monkeypatch):
    monkeypatch.setattr(metrics.time,'perf_counter_ns',lambda:1_000_000_000)
    row=metrics.build_terminal_attempt_metrics(Ctx(), fallback_force_calls=17)
    assert row['physical_pes_calls']==13
    assert row['physical_pes_calls_source']=='canonical_force_history.accounting.physical_total_pes_calls'
    assert row['legacy_n_force_calls']==17 and row['dimer_forcecalls']==17
    assert row['final_real_fmax']==0.04 and row['projected_fmax']==0.04
    assert row['projected_fmax_scope']=='last_accepted_dimer_translation'
    assert row['metric_availability']['pes_evaluation_wall_seconds']=='unavailable'
    assert row['pes_evaluation_wall_seconds'] is None
