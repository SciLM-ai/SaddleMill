#!/usr/bin/env python
"""arXiv SI figure (fig_si_signed_energy_err): stacked histograms of the signed UMA - DFT errors of the saddle energy,
endpoint energy (meV/atom) and barrier (meV) on the paper set, in the fig_main style (no title, shared legend, median beside its line).

Usage
-----
    python scripts/plot_signed_energy_err_si.py --root <.../paper_figs_20260926> [--outdir DIR]

Run after scripts/make_arxiv_figs.py: it reads <outdir>/base_set_ids.tsv written by that script (default outdir
<root>/arxiv_v20_insettext) and <root>/../barrier_err_20260926/{saddle_energies_per_case,barrier_err_per_point}.tsv.
This is the script that produced the arXiv SI figure (signed_hist_si_v20.py in that tree); only the arguments were added.
"""
import argparse, pandas as pd, numpy as np, pathlib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Patch
AP = argparse.ArgumentParser(); AP.add_argument("--root", required=True); AP.add_argument("--outdir", default=None); ARGS = AP.parse_args()
F = pathlib.Path(ARGS.root).resolve(); P = F.parent
OUT = pathlib.Path(ARGS.outdir).resolve() if ARGS.outdir else F/"arxiv_v20_insettext"
SPLITS = ["lemat", "mp20bat", "oc20", "oc22"]; COL = dict(zip(SPLITS, ["C0", "C1", "C2", "C3"]))
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11, "savefig.dpi": 300, "pdf.fonttype": 42})
base = pd.read_csv(OUT/"base_set_ids.tsv", sep="\t"); T = set(base[base.in_T_v13 == 1].ms_id.astype(int))
s = pd.read_csv(P/"barrier_err_20260926/saddle_energies_per_case.tsv", sep="\t", na_values=["NA"]); s = s[s.case_ms_id.astype(int).isin(T)].copy()
s["e"] = 1e3*(s.E_uma_s - s.E_dft_s)/s.natoms
b = pd.read_csv(P/"barrier_err_20260926/barrier_err_per_point.tsv", sep="\t"); b = b[b.case_ms_id.astype(int).isin(T) & (b.endpoint_status == "converged")].copy()  # v17: converged endpoints only
b["e_end"] = 1e3*(b.E_uma_end - b.E_dft_end)/b.natoms; b["e_bar"] = 1e3*b.err_eV
assert len(s) == 974 and len(b) == 1912
panels = [(s, "e", "Saddle: $E$(UMA) $-$ $E$(DFT) (meV/atom)", 20, 1.0, "meV/atom", ".2f"), (b, "e_end", "Endpoint: $E$(UMA) $-$ $E$(DFT) (meV/atom)", 20, 1.0, "meV/atom", ".2f"),
          (b, "e_bar", "Barrier: $E_\\mathrm{a}$(UMA) $-$ $E_\\mathrm{a}$(DFT) (meV)", 1000, 50.0, "meV", ".1f")]
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6), layout="constrained"); cap = []
for ax, (d, c, lab, lim, w, unit, fmt) in zip(axs, panels):
    bins = np.arange(-lim, lim + w, w); v = [d.loc[d.split == sp, c].clip(-lim + 1e-9, lim - 1e-9).values for sp in SPLITS]
    ax.hist(v, bins=bins, stacked=True, color=[COL[sp] for sp in SPLITS])
    allv = d[c].values; med = np.median(allv); out = np.mean(np.abs(allv) > lim)*100
    ax.set_ylim(0, 1.15*ax.get_ylim()[1])
    ax.axvline(0, color="k", lw=0.8); ax.axvline(med, color="k", ls="--", lw=1.5)
    ax.text(med + 0.02*lim, 0.97, f"median {med:+{fmt}} {unit}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=10)
    ax.set_xlabel(lab); ax.set_ylabel("count"); ax.set_xlim(-lim, lim); ax.set_box_aspect(1)
    cap.append(f"{c}: n={len(allv)}, median={med:+.2f} {unit}, beyond ±{lim:g} {unit} (drawn in the edge bins): {int(np.sum(np.abs(allv) > lim))} ({out:.1f}%)")
fig.legend(handles=[Patch(color=COL[sp], label=sp) for sp in SPLITS], loc="outside upper center", ncol=4, frameon=False)
for ext in ("png", "pdf"): fig.savefig(OUT/f"fig_si_signed_energy_err.{ext}", bbox_inches="tight", pad_inches=0.05)
print("\n".join(cap)); print("wrote", OUT/"fig_si_signed_energy_err.{png,pdf}")
