import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from ase.build import bulk, make_supercell
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator


predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")


atoms = bulk('C', 'diamond', a=3.57)
atoms = make_supercell(atoms, np.diag([2,2,2]))


def geomopt(i):
    atoms.positions += np.random.random((16,3))
    atoms.calc = calc

    opt = FIRE(FrechetCellFilter(atoms), logfile=f'optimization{i}.log')
    opt.run(0.0001, 1000)


if __name__ == "__main__":
    for i in range(3):
        geomopt(sys.argv[1])
