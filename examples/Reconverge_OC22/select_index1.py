"""select_index1.py <SinglePoint_trajes dir> <out dir> -- keep the reconverged saddles that are first order by the exact Hessian:
exactly one eigenvalue below -0.01 eV/A^2 and the next above +0.01 (for a structure without fixed atoms the three translations,
the eigenvalues closest to zero, are dropped first). Writes <out dir>/index1.traj for the double minimization."""
import glob, os, sys, numpy as np
from ase.io import Trajectory
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); n = k = 0
with Trajectory(f"{out}/index1.traj", "w") as w:
    for f in sorted(glob.glob(f"{src}/*.traj")):
        for a in Trajectory(f):
            ev = a.info.get("eigenvalues")
            if ev is None: continue
            n += 1; ev = np.sort(np.asarray(ev, float))
            if not any(type(c).__name__ == "FixAtoms" and len(c.get_indices()) for c in a.constraints):
                ev = np.delete(ev, np.argsort(np.abs(ev))[:3])
            if np.sum(ev < -0.01) == 1 and ev[1] > 0.01:
                a.info["approx_eigenmode"] = np.asarray(a.info["eigenmodes3"], float)[0]; a.info["approx_curvature"] = float(ev[0])
                w.write(a); k += 1
print(f"{k} of {n} reconverged saddles are index-1 -> {out}/index1.traj")
