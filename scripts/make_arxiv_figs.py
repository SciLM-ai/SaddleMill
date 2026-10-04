#!/usr/bin/env python
"""arXiv main figure (fig_main: saddle and endpoint max-atom-deviation histograms with energy-parity insets, barrier parity)
and the main table, from the UMA-vs-DFT endpoint campaign analysis tree.

Usage
-----
    python scripts/make_arxiv_figs.py --root <.../uma-endpoint-dft-2026-09-19/paper_figs_20260926> [--outdir DIR] [--base filtered|all]

``--root`` is the paper_figs directory of the analysis tree; its parent holds maxd_plot_20260926/, barrier_err_20260926/,
saddle_steps_20260927/, spin_flags_20260926/, potcar_redo_spin_audit_20260930/, peer_handoff_20261003/ and
production_inputs_manifest.tsv. Outputs go to ``--outdir`` (default <root>/arxiv_v20_insettext): fig_main.{png,pdf},
fig_main_{saddle,endpoint}_maxd.{png,pdf}, table_main.{tex,tsv}, base_set_ids.tsv. This is the script that produced the
arXiv fig_main and table (make_arxiv_figs_v20.py in that tree); only the ``--root`` argument was added.

Original spec (sung, 2026-09-27): one row of three square panels
(saddle max-d histogram, endpoint max-d histogram, barrier parity plot), one shared split legend, no titles,
no p90 lines, no spin flags, no n (stated in the caption); median as a black dashed line with its value beside it,
> 0.2 A fraction beside the red line, and the parity MAE / RMSE / |error| > 0.1 eV fraction as in-panel text; beyond-axes points hollow at
the edge (count in the caption). Table: one transposed table (rows = metrics in three groups: Saddle, Endpoint, Barrier;
columns = splits + all); convergence as converged / finished, energies in meV and meV/atom; running jobs never counted. Same analysis set T as make_paper_figs.py
(DFT saddle converged, DFT and UMA saddle energies present), cross-checked against count_identity.tsv."""
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator  # v18

AP = argparse.ArgumentParser()
AP.add_argument("--root", required=True, help="paper_figs_20260926 directory of the analysis tree (see module docstring)")
AP.add_argument("--base", choices=["all", "filtered"], default="filtered")
AP.add_argument("--outdir", default=None, help="default <root>/arxiv_v20_insettext")
ARGS = AP.parse_args()
HERE = pathlib.Path(ARGS.root).resolve()
W = HERE.parent
MD, BE = W / "maxd_plot_20260926", W / "barrier_err_20260926"
AUD = BE / "dft_saddle_geometry_source_audit.tsv"
SS = W / "saddle_steps_20260927" / "saddle_steps_per_case_v2.tsv"  # dimer steps per case (DIMCAR max Step, banked + own)
POT = W / "saddle_steps_20260927" / "potcar_mixed_library_ids.tsv"  # pass 13: 72 redo dimers folded in (campaign PBE_54 library, potcar_redo_20260927); 101825890 excluded as uma_false_positive
POT_IDS = set(pd.read_csv(POT, sep="\t").ms_id.astype(int))  # int ids, same typing as case_ms_id below
POT_REASON = ",".join(sorted(set(pd.read_csv(POT, sep="\t").reason.astype(str))))  # pass 13: label from the file's reason column
SPIN = W / "spin_flags_20260926" / "endpoint_spin_flags.tsv"  # pass 13: same endpoint flags as make_paper_figs.py
SADF = W / "potcar_redo_spin_audit_20260930" / "redo_saddle_flags.tsv"  # pass 13: saddle-level flags
EP_FLAGGED, SADFLAGGED = {"SPIN_PATH"}, {"SPIN", "SPIN_PATH", "SPIN_DEGENERATE"}  # pass 13: same sets as make_paper_figs.py
# v14 (2026-10-03, sung): energy parity (eV/atom, UMA vs DFT) as an inset in the saddle and endpoint histogram panels; everything else = v13.
# v15 (2026-10-03, sung): table only. Saddle "converged / total (filtered)", no saddle n row; Endpoint "converged / total" = converged / 2 x n_saddle on T (unfinished and capped = not converged); endpoint steps over the T-converged endpoints. Figures = v14.
# v16 (2026-10-03, sung): fig_main histogram x axis 1e-4..1e1 (bins extended, same 15 per decade); parity inset moved to the upper left with its y axis on the left; median label raised to the > 0.2 A label level. Table = v15.
# v17 (2026-10-03, sung): endpoint histogram, inset, barrier panel, SI histograms and all Endpoint/Barrier table rows use CONVERGED endpoints only (1912 = 2 x 974 - 31 unfinished - 5 capped); Endpoint n row dropped (= converged).
# v18 (2026-10-03, sung): inset ticks every 5 eV/atom on both axes; inset moved right (x 0.10 -> 0.14, width 0.35 -> 0.32) so its y label clears the outer y axis. Data = v17.
# v19 (2026-10-03, sung): inset maximised: largest square clearing the bars, the median label/line and the frame (renderer-measured clearances, scratchpad v19/harness_v19.py) = [0.01, 0.52, 0.36, 0.36] with the y axis on the right; 0.39 put the right-side y label on the saddle bars. Data = v17.
# v20 (2026-10-03, sung): inset annotation RMSE on top, MAE below, font 8.5 pt (was 6.5). Data = v17.
# v13 (2026-10-03): --base filtered restricts every row to the paper base B (F1, F2, F3b all not True; q6_paper_filters).
FLT = ARGS.base == "filtered"
OUT = pathlib.Path(ARGS.outdir).resolve() if ARGS.outdir else HERE / "arxiv_v20_insettext"
assert OUT.name not in ("arxiv", "arxiv_v13_filtered", "arxiv_v14_insets", "arxiv_v15_table", "arxiv_v16_insetleft", "arxiv_v17_converged", "arxiv_v18_insetticks", "arxiv_v19_insetmax"), "v20 never writes into the v12-v19 dirs"
OUT.mkdir(parents=True, exist_ok=True)
FLAGS = W / "peer_handoff_20261003" / "q6_paper_filters" / "step1_per_saddle_flags.tsv"
MAN = W / "production_inputs_manifest.tsv"  # running endpoints, same rule as make_paper_figs.py
NOSPIN_PATH = HERE / "table_main_nospin_v20.tsv" if FLT else OUT / "table_main_nospin.tsv"
IDENT_PATH = HERE / "count_identity_v20.tsv" if FLT else OUT / "count_identity.tsv"
B_EXPECT = dict(lemat=353, mp20bat=181, oc20=273, oc22=204)
TB_EXPECT = dict(lemat=348, mp20bat=172, oc20=261, oc22=193)
SPLITS = ["lemat", "mp20bat", "oc20", "oc22"]
COL = dict(zip(SPLITS, ["C0", "C1", "C2", "C3"]))
LOGBINS = np.logspace(-4, 1, 76)  # v16: 1e-4..1e1, 15 bins per decade as before
PLIM = (-0.5, 8.0)  # parity axes (eV); points beyond are drawn hollow at the edge and counted in the panel text
TXT = dict(ha="left", va="top", fontsize=10)
INSET_BOX = [0.01, 0.52, 0.36, 0.36]  # v19 parity inset (axes fraction): flush with the outer left spine, y axis on the right
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 11, "savefig.dpi": 300, "pdf.fonttype": 42})


def analysis_set(au):
    unconv = set(au.loc[au.converged.isin(["0", "0.0"]), "case_ms_id"])
    no_dft = set(au.loc[au.dft_s_src_used == "NA", "case_ms_id"])
    no_uma = set(au.loc[au.uma_s_src_used == "NA", "case_ms_id"])
    return set(au.case_ms_id) - unconv - no_dft - no_uma - POT_IDS


def load_base():
    fl = pd.read_csv(FLAGS, sep="\t")
    print("FLAGS dtypes", {c: str(fl[c].dtype) for c in ("ms", "split", "F1", "F2", "F3b")}, "rows", len(fl))
    assert len(fl) == 1560 and list(fl.columns[[0, 42, 43, 44, 45]]) == ["ms", "split", "F1", "F2", "F3b"]
    keep = ~(fl.F1.astype(str) == "True") & ~(fl.F2.astype(str) == "True") & ~(fl.F3b.astype(str) == "True")
    b = fl[keep]
    cnt = b.split.value_counts().to_dict()
    print("ASSERT B", len(b), cnt)
    assert len(b) == 1011 and cnt == B_EXPECT, (len(b), cnt)
    assert b.ms.is_unique
    return b[["ms", "split"]].astype({"ms": int})


def maxd_stats(x):
    x = np.asarray(x, float)
    return dict(n=len(x), median=np.median(x), mean=x.mean(), p90=np.percentile(x, 90), frac_gt_0p2=np.mean(x > 0.2))


def err_stats(e):
    e = np.asarray(e, float)
    a = np.abs(e)
    return dict(n=len(e), mae=a.mean(), rmse=np.sqrt(np.mean(e ** 2)), median_err=np.median(e), mean_err=e.mean(),
                frac_abs_gt_0p1=np.mean(a > 0.1), frac_abs_gt_0p2=np.mean(a > 0.2))


def hist_panel(ax, df, xlabel, inset=None):
    ax.hist([df.loc[df.split == sp, "maxd_A"] for sp in SPLITS], bins=LOGBINS, stacked=True,
            color=[COL[sp] for sp in SPLITS])
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e1)  # v16
    ax.axvline(0.2, color="red", ls="--", lw=1.5)
    s = maxd_stats(df.maxd_A)
    ax.set_ylim(0, (1.5 if inset is not None else 1.22) * ax.get_ylim()[1])  # headroom for the two labels (+ inset)
    ax.axvline(s["median"], color="k", ls="--", lw=1.5)
    ax.text(s["median"] / 1.08, 0.97, f"median {s['median']:.3f} Å", transform=ax.get_xaxis_transform(),  # v16: same level as the > 0.2 A label
            ha="right", va="top", fontsize=10)
    ax.text(0.2 * 1.08, 0.97, f"> 0.2 Å: {100 * s['frac_gt_0p2']:.1f}%", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_box_aspect(1)
    if inset is not None:  # v14: energy parity inset, eV/atom, UMA vs DFT, coloured by split
        x, y, spl = (np.asarray(v) for v in inset)
        ia = ax.inset_axes(INSET_BOX)
        for sp in SPLITS:
            m = spl == sp
            ia.scatter(x[m], y[m], s=4, color=COL[sp], alpha=0.6, lw=0)
        lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
        ia.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8)
        ia.set_xlim(lo, hi); ia.set_ylim(lo, hi); ia.set_box_aspect(1)
        ia.yaxis.tick_right(); ia.yaxis.set_label_position("right")  # v19: y axis on the right, box flush with the outer left spine
        ia.tick_params(labelsize=7, length=2, pad=1)
        ia.xaxis.set_major_locator(MultipleLocator(5)); ia.yaxis.set_major_locator(MultipleLocator(5))  # v18: -10, -5, 0 on both axes
        ia.set_xlabel("$E$(DFT) (eV/atom)", fontsize=7.5, labelpad=1); ia.set_ylabel("$E$(UMA) (eV/atom)", fontsize=7.5, labelpad=1)
        e = 1e3 * (y - x); a = np.abs(e)
        s.update(e_n=len(e), e_mae_meV_atom=a.mean(), e_rmse_meV_atom=np.sqrt(np.mean(e ** 2)), e_frac_abs_gt_10=np.mean(a > 10))
        ia.text(0.04, 0.97, f"RMSE {np.sqrt(np.mean(e ** 2)):.1f}\nMAE {a.mean():.1f}\n(meV/atom)",  # v20: RMSE on top, MAE below, unit on its own line, 8.5 pt; upper-left triangle, clear of the diagonal
                transform=ia.transAxes, ha="left", va="top", fontsize=8.5, linespacing=1.15)
    return s


def parity_panel(ax, bar, color=None, clabel=None, fig=None):
    """color=None: points colored by split. Otherwise `color` = per-row max-atom deviation (A), mapped on the same
    log scale as the histogram x axes (1e-3..1e1, viridis, red dashed 0.2 A mark on the colorbar), largest drawn last."""
    lo, hi = PLIM
    x, y = bar.Ea_dft.to_numpy(float), bar.Ea_uma.to_numpy(float)
    out = (x < lo) | (x > hi) | (y < lo) | (y > hi)
    if color is None:
        for sp in SPLITS:
            m = (bar.split == sp).to_numpy()
            ax.scatter(x[m & ~out], y[m & ~out], s=9, color=COL[sp], alpha=0.6, lw=0)
            if (m & out).any():
                ax.scatter(np.clip(x[m & out], lo, hi), np.clip(y[m & out], lo, hi), s=24, facecolors="none",
                           edgecolors=COL[sp], lw=0.9)
    else:
        c = np.clip(np.asarray(color, float), 1e-3, 1e1)
        assert np.isfinite(c).all() and len(c) == len(x)
        order = np.argsort(c)
        inn, oo = order[~out[order]], order[out[order]]
        norm = LogNorm(1e-3, 1e1)
        sc = ax.scatter(x[inn], y[inn], c=c[inn], s=9, cmap="viridis", norm=norm, alpha=0.8, lw=0)
        if len(oo):
            ax.scatter(np.clip(x[oo], lo, hi), np.clip(y[oo], lo, hi), s=24, facecolors="none",
                       edgecolors=plt.cm.viridis(norm(c[oo])), lw=0.9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clabel)
        cb.ax.axhline(0.2, color="red", ls="--", lw=1.5)
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$E_\mathrm{a}$(DFT) (eV)")
    ax.set_ylabel(r"$E_\mathrm{a}$(UMA) (eV)")
    s = err_stats(bar.err_eV)
    s["n_outside_axes"] = int(out.sum())
    ax.text(0.03, 0.97, f"MAE = {1e3 * s['mae']:.0f} meV\nRMSE = {1e3 * s['rmse']:.0f} meV\n|error| > 100 meV: {100 * s['frac_abs_gt_0p1']:.1f}%",
            transform=ax.transAxes, **TXT)
    return s


def main():
    au = pd.read_csv(AUD, sep="\t", keep_default_na=False, dtype=str)
    au["case_ms_id"] = au.case_ms_id.astype(int)
    T = analysis_set(au)
    ident = pd.read_csv(HERE / "count_identity.tsv", sep="\t")
    pooled = ident[ident.split == "pooled"].iloc[0]
    assert len(T) == int(pooled.n_saddle), ("T differs from make_paper_figs.py", len(T), pooled.n_saddle)
    sad = pd.read_csv(MD / "saddle_maxd_per_triplet.tsv", sep="\t")
    end = pd.read_csv(MD / "maxd_per_endpoint.tsv", sep="\t")
    bar = pd.read_csv(BE / "barrier_err_per_point.tsv", sep="\t")
    sadT, endT, barT = sad[sad.saddle_ms_id.isin(T)], end[end.case_ms_id.isin(T)], bar[bar.case_ms_id.isin(T)]
    assert (len(endT), len(barT)) == (int(pooled.n_endpoint), int(pooled.n_barrier)), (len(endT), len(barT))
    if FLT:  # v13: T_v13 = T & B; the refined (au) and finished (end) frames restricted to cases in B
        bdf = load_base()
        B = set(bdf.ms)
        assert B <= set(au.case_ms_id), "B ids missing from the audit"
        assert dict(zip(bdf.ms, bdf.split)) == {c: s for c, s in zip(au.case_ms_id, au.split) if c in B}, "split mismatch B vs audit"
        T = T & B
        au, end = au[au.case_ms_id.isin(B)], end[end.case_ms_id.isin(B)]
        sadT, endT, barT = sad[sad.saddle_ms_id.isin(T)], end[end.case_ms_id.isin(T) & (end.status == "converged")], bar[bar.case_ms_id.isin(T) & (bar.endpoint_status == "converged")]  # v17: converged endpoints only
        tcnt = sadT.split.value_counts().to_dict()
        print("ASSERT T_v13", len(T), len(sadT), tcnt)
        assert len(T) == 974 == len(sadT) and tcnt == TB_EXPECT, (len(T), tcnt)
        ids = bdf.rename(columns={"ms": "ms_id"}).assign(in_T_v13=lambda d: d.ms_id.isin(T).astype(int))
        ids.to_csv(OUT / "base_set_ids.tsv", sep="\t", index=False)
    assert barT.err_eV.notna().all() and endT.maxd_A.notna().all() and sadT.maxd_A.notna().all()
    print(f"T={len(T)} saddle={len(sadT)} endpoint={len(endT)} barrier={len(barT)}")

    # per-barrier-row colour keys for the two variants: the case's saddle max-d, and the endpoint's own max-d
    sad_maxd = sadT.set_index("saddle_ms_id").maxd_A.reindex(barT.case_ms_id).to_numpy(float)
    end_map = endT.set_index(["ms_id", "role"]).maxd_A
    end_maxd = end_map.reindex(list(zip(barT.endpoint_ms_id, barT.role))).to_numpy(float)
    assert np.isfinite(sad_maxd).all() and np.isfinite(end_maxd).all()
    assert np.allclose(end_maxd, barT.maxd_A.to_numpy(float), atol=1e-6), "barrier-table maxd_A != endpoint-table maxd_A"
    sE = pd.read_csv(BE / "saddle_energies_per_case.tsv", sep="\t", na_values=["NA"])  # v14 insets: saddle energies per case, endpoint energies from the barrier rows
    sE = sE.set_index("case_ms_id").reindex(sadT.saddle_ms_id.to_numpy())
    assert len(sE) == len(sadT) and sE.E_uma_s.notna().all() and sE.E_dft_s.notna().all() and (sE.split.to_numpy() == sadT.split.to_numpy()).all()
    assert len(barT) == len(endT) and barT.E_uma_end.notna().all() and barT.E_dft_end.notna().all()
    ins_sad = ((sE.E_dft_s / sE.natoms).to_numpy(float), (sE.E_uma_s / sE.natoms).to_numpy(float), sE.split.to_numpy())
    ins_end = ((barT.E_dft_end / barT.natoms).to_numpy(float), (barT.E_uma_end / barT.natoms).to_numpy(float), barT.split.to_numpy())
    for name, color, clabel in (("fig_main", None, None),
                                ("fig_main_saddle_maxd", sad_maxd, "saddle max-atom deviation (Å)"),
                                ("fig_main_endpoint_maxd", end_maxd, "endpoint max-atom deviation (Å)")):
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6), layout="constrained")
        s_sad = hist_panel(axs[0], sadT, "Saddle: UMA vs DFT max-atom deviation (Å)", inset=ins_sad)
        s_end = hist_panel(axs[1], endT, "Endpoint: UMA vs DFT max-atom deviation (Å)", inset=ins_end)
        s_bar = parity_panel(axs[2], barT, color=color, clabel=clabel, fig=fig)
        fig.legend(handles=[Patch(color=COL[sp], label=sp) for sp in SPLITS], loc="outside upper center", ncol=4, frameon=False)
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    print("POOLED", {k: round(float(v), 4) for k, v in {**{"sad_" + k: v for k, v in s_sad.items()},
                                                          **{"end_" + k: v for k, v in s_end.items()},
                                                          **{"bar_" + k: v for k, v in s_bar.items()}}.items()})

    def sel(d, sp):
        return d if sp == "all" else d[d.split == sp]

    # pass 13: nospin masks (endpoint flag in EP_FLAGGED or saddle flag in SADFLAGGED), same rule as make_paper_figs.py
    sf = pd.read_csv(SPIN, sep="\t", dtype={"ms_id": str, "role": str})
    epf = {(int(a), b) for a, b, c in zip(sf.ms_id, sf.role, sf.spin_flag) if c in EP_FLAGGED}
    sfl = pd.read_csv(SADF, sep="\t", dtype={"ms_id": int})
    sadf = set(sfl.loc[sfl.flag.isin(SADFLAGGED), "ms_id"])
    epflag = lambda d: pd.Series([(int(a), b) in epf for a, b in zip(d.ms_id, d.role)], index=d.index, dtype=bool)
    rowflag = lambda d, cm: epflag(d) | d[cm].astype(int).isin(sadf)  # endpoint-table rows (ms_id, role, cm = case id column)
    flag_cases = sadf | {int(c) for c, f in zip(endT.case_ms_id, epflag(endT)) if f}
    nsp = dict(sad=sadT[~sadT.saddle_ms_id.astype(int).isin(sadf)],  # pass 13b: saddle nospin = saddle flag only
               end=endT[~rowflag(endT, "case_ms_id")],
               bar=barT[~pd.Series([(int(a), b) in epf for a, b in zip(barT.endpoint_ms_id, barT.role)], index=barT.index, dtype=bool)
                        & ~barT.case_ms_id.astype(int).isin(sadf)])
    man = pd.read_csv(MAN, sep="\t")  # v13: count identity, same columns and rules as make_paper_figs.py
    run = man[man.case_ms_id.isin(T) & ~man.requested_ms_id.isin(set(end.ms_id.astype(int)))]
    cap = end[end.case_ms_id.isin(T) & (end.status != "converged")]  # v17: finished but not converged (step-capped)
    ef, bf = rowflag(endT, "case_ms_id"), pd.Series([(int(a), b) in epf for a, b in zip(barT.endpoint_ms_id, barT.role)], index=barT.index, dtype=bool) | barT.case_ms_id.astype(int).isin(sadf)
    idv = []
    for sp in SPLITS + ["pooled"]:
        m = (lambda d: d if sp == "pooled" else d[d.split == sp])
        fe, fb, fs_ = int(ef[m(endT).index].sum()), int(bf[m(barT).index].sum()), int(m(sadT).saddle_ms_id.astype(int).isin(sadf).sum())
        idv.append(dict(split=sp, n_saddle=len(m(sadT)), n_endpoint=len(m(endT)), n_running=len(m(run)), n_capped=len(m(cap)), n_barrier=len(m(barT)),
                        two_saddle_minus_running_capped=2 * len(m(sadT)) - len(m(run)) - len(m(cap)), n_flagged=fe, n_endpoint_nospin=len(m(endT)) - fe,
                        n_barrier_nospin=len(m(barT)) - fb, n_saddle_flagged=fs_, n_saddle_nospin=len(m(sadT)) - fs_))
    idv = pd.DataFrame(idv)
    idv["endpoint_ok"] = idv.n_endpoint == idv.two_saddle_minus_running_capped
    idv["barrier_ok"] = (idv.n_barrier == idv.n_endpoint) & (idv.n_barrier_nospin == idv.n_endpoint_nospin)
    idv.to_csv(IDENT_PATH, sep="\t", index=False)
    print("IDENTITY_V13 ->", IDENT_PATH, "\n" + idv.to_string(index=False))
    assert idv.endpoint_ok.all() and idv.barrier_ok.all(), "count identities violated"
    if not FLT:
        assert idv.equals(ident), "recomputed count identity != count_identity.tsv"
        print("IDENTITY_V13 == count_identity.tsv: True")
    else:
        ident = idv
    nsp_id = ident[ident.split == "pooled"].iloc[0]
    assert (len(nsp["sad"]), len(nsp["end"]), len(nsp["bar"])) == (int(nsp_id.n_saddle_nospin), int(nsp_id.n_endpoint_nospin), int(nsp_id.n_barrier_nospin)), \
        ("nospin differs from count_identity.tsv", len(nsp["sad"]), len(nsp["end"]), len(nsp["bar"]))
    print(f"NOSPIN saddle={len(nsp['sad'])} endpoint={len(nsp['end'])} barrier={len(nsp['bar'])} flagged_saddle_cases_inT={len(sadf & T)}")
    sadE_all = pd.read_csv(BE / "saddle_energies_per_case.tsv", sep="\t", na_values=["NA"])  # pass 13
    ss_all = pd.read_csv(SS, sep="\t", na_values=["NA"])  # pass 13

    def build(sadT, endT, barT, au, end, Tset, flags=False):  # pass 13: table body as a function of the row subset
        sadE = sadE_all
        sadE = sadE[sadE.case_ms_id.isin(Tset)]
        assert len(sadE) == len(Tset) and sadE.E_uma_s.notna().all() and sadE.E_dft_s.notna().all()
        sadE = sadE.assign(dE_atom=(sadE.E_uma_s - sadE.E_dft_s) / sadE.natoms)
        barT = barT.assign(dE_end_atom=(barT.E_uma_end - barT.E_dft_end) / barT.natoms)
        endC = endT  # v17: endT is converged-only

        rows = []  # (group, tex label, tsv key, formatted {split: cell})
        def add(group, tex, key, fmt, fn):
            rows.append((group, tex, key, {sp: fmt(fn(sp)) for sp in SPLITS + ["all"]}))
        f0, f1, f3 = (lambda v: f"{v:.0f}"), (lambda v: f"{v:.1f}"), (lambda v: f"{v:.3f}")
        pct = lambda v: f"{100 * v:.1f}"
        ratio = lambda v: f"{v[0]:.0f} / {v[1]:.0f}"
        add("Saddle", "converged / total (filtered)" if FLT else "converged / refined", "converged_over_refined", ratio,
            lambda sp: ((~sel(au, sp).converged.isin(["0", "0.0"]) & ~sel(au, sp).case_ms_id.isin(POT_IDS)).sum(), len(sel(au, sp))))  # pass 14b: POT-excluded cases are not converged for the paper
        add("Saddle", "median max-atom deviation (\\AA)", "median_maxd_A", f3, lambda sp: sel(sadT, sp).maxd_A.median())
        add("Saddle", "$>0.2$ \\AA\\ (\\%)", "frac_gt_0p2", pct, lambda sp: (sel(sadT, sp).maxd_A > 0.2).mean())
        add("Saddle", "energy MAE (meV/atom)", "energy_mae_meV_per_atom", f1, lambda sp: 1e3 * sel(sadE, sp).dE_atom.abs().mean())
        ss = ss_all
        ss = ss[ss.case_ms_id.isin(Tset) & ss.cumulative_best.notna()]
        print("SADDLE_STEPS", {sp: len(sel(ss, sp)) for sp in SPLITS + ["all"]}, "of T", len(Tset),
              "banking", ss.banking_flag.value_counts().to_dict())
        add("Saddle", "steps to converge (median)", "steps_median", f0, lambda sp: sel(ss, sp).cumulative_best.median())
        add("Endpoint", "converged / total", "converged_over_total", ratio,
            lambda sp: (len(sel(endC, sp)), 2 * len(sel(sadT, sp))))  # v15: total = 2 endpoints per T saddle; unfinished + capped = not converged
        add("Endpoint", "median max-atom deviation (\\AA)", "median_maxd_A", f3, lambda sp: sel(endT, sp).maxd_A.median())
        add("Endpoint", "$>0.2$ \\AA\\ (\\%)", "frac_gt_0p2", pct, lambda sp: (sel(endT, sp).maxd_A > 0.2).mean())
        add("Endpoint", "energy MAE (meV/atom)", "energy_mae_meV_per_atom", f1, lambda sp: 1e3 * sel(barT, sp).dE_end_atom.abs().mean())
        add("Endpoint", "steps to converge (median)", "steps_median", f0, lambda sp: sel(endC, sp).steps.median())
        add("Barrier", "MAE (meV)", "mae_meV", f0, lambda sp: 1e3 * sel(barT, sp).err_eV.abs().mean())
        add("Barrier", "RMSE (meV)", "rmse_meV", f0, lambda sp: 1e3 * np.sqrt((sel(barT, sp).err_eV ** 2).mean()))
        add("Barrier", "$|$error$| > 100$ meV (\\%)", "frac_abs_gt_100meV", pct, lambda sp: (sel(barT, sp).err_eV.abs() > 0.1).mean())
        if flags:  # pass 13
            add("Flags", "flagged saddle cases", "n_flagged_saddle_cases", f0, lambda sp: len(sel(sadT0[sadT0.saddle_ms_id.astype(int).isin(sadf)], sp)))
            add("Flags", "endpoint rows excluded (own SPIN_PATH or saddle flag)", "n_flagged_endpoint_rows", f0, lambda sp: len(sel(endT0, sp)) - len(sel(endT, sp)))  # pass 13b
        return rows, ss

    sadT0 = sadT  # pass 13
    endT0 = endT  # pass 13b
    rows_ns, _ = build(nsp["sad"], nsp["end"], nsp["bar"], au[~au.case_ms_id.isin(sadf)], end[~rowflag(end, "case_ms_id")],
                       T - sadf, flags=True)  # pass 13b: saddle rows on T minus saddle-flagged cases
    RULE = {"Saddle": "saddle_flag_only", "Endpoint": "own_SPIN_PATH_or_saddle_flag", "Barrier": "own_SPIN_PATH_or_saddle_flag"}  # pass 13b
    RULE_FLAG = {"n_flagged_saddle_cases": "saddle_flag_only", "n_flagged_endpoint_rows": "own_SPIN_PATH_or_saddle_flag"}  # pass 13b
    tab_ns = pd.DataFrame([dict(group=g, metric=k, **v, nospin_rule=RULE.get(g, RULE_FLAG.get(k))) for g, _, k, v in rows_ns])  # pass 13b: rule column
    tab_ns.to_csv(NOSPIN_PATH, sep="\t", index=False)  # pass 13; pass 14: written to paper_figs_20260926/, not arxiv/ (sung 2026-10-03: no spin-flag labels in the arXiv deliverables)
    print("TABLE_NOSPIN\n" + tab_ns.to_string(index=False))  # pass 13
    rows, ss = build(sadT, endT, barT, au, end, T)  # pass 13: the "all" table, unchanged
    tab = pd.DataFrame([dict(group=g, metric=k, **v) for g, _, k, v in rows])
    tab.to_csv(OUT / "table_main.tsv", sep="\t", index=False)
    cols = SPLITS + ["all"]
    lines = [r"% needs \usepackage{booktabs,multirow}", r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             " & & " + " & ".join(cols) + r" \\", r"\midrule"]
    for gi, g in enumerate(dict.fromkeys(r[0] for r in rows)):
        grp = [r for r in rows if r[0] == g]
        if gi:
            lines.append(r"\midrule")
        for i, (_, tex, _, v) in enumerate(grp):
            lab = rf"\multirow{{{len(grp)}}}{{*}}{{{g}}}" if i == 0 else ""
            lines.append(f"{lab} & {tex} & " + " & ".join(v[sp] for sp in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "% converged / refined: DFT saddle refinements over all 1622 cases (all finished); converged / finished: DFT endpoint relaxations, running jobs excluded." if not FLT else
              f"% converged / total (filtered): triplets with a converged DFT saddle over the {len(au)} triplets passing the three UMA filters (endpoint collapse, non-positive barrier, mode rule); converged / total: converged DFT endpoint relaxations over the 2 x 974 requested for the T triplets (unfinished and step-capped relaxations count as not converged).",
              f"% median, > 0.2 A, energy MAE: analysis set T ({'triplets passing the three UMA filters (endpoint collapse, non-positive barrier, mode rule) with a converged DFT saddle; ' if FLT else ''}dimer-converged DFT saddle with DFT and UMA saddle energies; DFT saddle energy = the dimer's final single point; {', '.join(f'{v} {k}' for k, v in (lambda p: p[p.ms_id.astype(int).isin(set(au.case_ms_id))] if FLT else p)(pd.read_csv(POT, sep='\t')).reason.astype(str).value_counts().sort_index().items())} case(s) excluded; {(lambda r: int(r.ms_id.astype(int).isin(set(au.case_ms_id)).sum()) if FLT else len(r))(pd.read_csv(W / 'potcar_redo_spin_audit_20260930' / 'redo_override.tsv', sep='\t'))} POTCAR-redo dimers on the campaign PBE_54 library); max-atom deviation and |E(UMA) - E(DFT)| / natoms, UMA vs DFT saddle or endpoint.",
              f"% steps to converge: saddle = dimer translation steps (DIMCAR max step, cumulative over restarts) for {len(ss)} of the {len(T)} T cases; endpoint = DFT relaxation steps over the converged T endpoints. Endpoint and Barrier rows, the endpoint histogram and inset, and the barrier panel use the converged endpoints of the T triplets only (unfinished and step-capped excluded). Barrier: Ea(UMA) - Ea(DFT) per converged endpoint."]
    (OUT / "table_main.tex").write_text("\n".join(lines) + "\n")
    print(tab.to_string(index=False))

if __name__ == "__main__":
    main()
