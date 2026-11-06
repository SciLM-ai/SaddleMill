# Reconverge a dataset's saddle labels (OC22)

Tighten every saddle label of a dataset to a verified first-order saddle and give it its own minima, with UMA-S-1.2.
This is the pipeline that produced the reconverged OC22 and OC20 training sets for SaddleFlow (November 2026).

| stage | directory | method | settings |
|---|---|---|---|
| 1 | `1_sella` | Sella from each label | first order, Cartesian, `delta0 0.1`, `eta 1e-3`, **fmax 0.005**, <= 500 steps |
| 2 | `2_hessian` | SinglePoint, exact Hessian | `compute_eigenmodes`: the UMA Hessian over the free atoms, its spectrum and lowest mode |
| - | `select_index1.py` | keep first-order saddles | exactly one eigenvalue < -0.01 eV/A^2 and the next > +0.01 (translations dropped when no atom is fixed) |
| 3 | `3_doubleminimization` | DoubleMinimization | FIRE along the exact lowest mode, the DoubleMinimization_OC22 settings, **fmax 0.005**, <= 1000 steps |

Put the labels (one frame per saddle, with the slab's `FixAtoms`; any `info` you want carried through, e.g. a triplet id) in
`data/` and run `run_vista.sh`, or run the stages one by one (`cd 1_sella && flux start python -u -m saddlemill`, ...).
A training triplet is then `(m-, S', m+)`: the reconverged saddle and the two minima of its double minimization. Keep it when
the three structures are pairwise more than 0.2 A apart (max-atom, minimum image) and both barriers are positive.

## Run settings that matter

- **One CPU math thread per job** (`[Main] threads_per_job = 1`, the default). Sella works in the full 3N Cartesian space
  (fixed atoms are constraints), so on a 225-atom slab every step runs `eigh`, `lstsq` and QR on 675 x 675 matrices on the CPU.
  With multithreaded BLAS in every worker, 8 such jobs took 679 s; single-threaded, 66 s, and the GPU stopped idling.
- **Sella's step eigenproblem on the GPU** (`[ourSella] gpu_eigh = True`, the default; needs the GPU-enabled Sella).
- **`eta = 0.001`.** Sella's default 1e-4 finite-difference step is below UMA's GPU force noise and can stall its mode search.
- **Workers per GPU by cell size.** 16 Sella / FIRE jobs per GPU fit for cells up to ~90 atoms; on larger cells scale down
  (jobs x largest cell <= ~1200-1600 atoms per instance) or the GPU runs out of memory. The exact Hessian is GPU-bound and
  memory-heavy: 2 jobs per instance. A GPU running one Hessian instance and one Sella/FIRE instance does more total work than
  two of either (the first is GPU-bound, the second CPU-bound).
- **Same potential throughout**, with the subset's own task (`oc22` here).

## Shortcuts that were measured before use (OC22, 132,504 reconverged saddles)

- **Skip the double minimization when the saddle barely moved.** When the reconverged saddle is within 0.1 A of its label, its
  FIRE minima are the label's own reactant and product in 92-99 % of cases (0-0.05 A: 98.5 %; 0.05-0.1 A: 91.6 %, within
  0.1 A); continuing those endpoints to fmax 0.005 (`method = Minimization`) costs far less than minimizing from the saddle.
- **Defer the Hessian for already-verified labels.** For a label that was already first order by this same criterion (the
  MaterialsSaddles v4 "main" files) and moved < 0.2 A, the reconverged saddle stayed index-1 in 99.6 % of cases (< 0.1 A:
  99.7 %, 0.1-0.2 A: 99.0 %). Not for flagged ("tier2") labels: 85 %. Mark such saddles as Hessian-pending and store no
  Hessian fields for them until it is computed.
