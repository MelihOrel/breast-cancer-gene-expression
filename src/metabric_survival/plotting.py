"""Figures (300 dpi PNG). One visual system for every chart.

Colour roles follow a validated palette: categorical slots in a fixed order for
nominal groups, a single-hue ramp for ordered groups (age bands, NPI groups),
a blue/red pair for protective vs harmful effects. Text always uses ink tones,
never a series colour. Every figure has a table twin in results/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]
PROTECTIVE, HARMFUL = "#2a78d6", "#e34948"
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

MODEL_LABELS = {"coxnet": "Elastic-net Cox", "rsf": "Random survival forest", "xgb_cox": "Gradient-boosted Cox"}

TITLE_IN, SUBTITLE_IN = 0.08, 0.36      # distance from the top edge, inches
HEAD_IN = 0.78                          # space reserved for title + subtitle, inches


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    })


def _top(fig_h: float, extra_in: float = 0.0) -> float:
    """Axes top (figure fraction) leaving room for the heading plus ``extra_in`` inches."""
    return 1 - (HEAD_IN + extra_in) / fig_h


def _heading(fig: plt.Figure, title: str, subtitle: str | None = None, x: float = 0.02) -> None:
    h = fig.get_figheight()
    fig.text(x, 1 - TITLE_IN / h, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 1 - SUBTITLE_IN / h, subtitle, ha="left", va="top", fontsize=8.5, color=INK_2)


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def _signed(v: float, digits: int = 3) -> str:
    v = round(float(v), digits) + 0.0          # no "-0.000"
    return f"{v:+.{digits}f}" if v != 0 else f"{0:.{digits}f}"


def _rounded_hbar(ax, y: float, x0: float, x1: float, height: float, color: str, radius_pt: float = 3.0) -> None:
    """Horizontal bar from baseline x0 to data end x1: rounded data end, square baseline.

    Axis limits and figure layout must be final before calling (the corner radius is
    converted from points to data units in x and y separately so it stays circular).
    """
    if x1 == x0:
        return
    bbox = ax.get_window_extent()
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    px = radius_pt * ax.figure.dpi / 72
    rx = px * abs(xb - xa) / bbox.width
    ry = px * abs(yb - ya) / bbox.height
    w = abs(x1 - x0)
    r = min(rx, w)
    ax.add_patch(FancyBboxPatch((min(x0, x1), y - height / 2), w, height,
                                boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=ry / rx,
                                facecolor=color, edgecolor="none", zorder=3))
    sq_left = x0 if x1 > x0 else x0 - r
    ax.add_patch(Rectangle((sq_left, y - height / 2), r, height, facecolor=color, edgecolor="none", zorder=3))


def _years_axis(ax, max_months: float) -> None:
    ticks = np.arange(0, max_months + 1, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t / 12)}" for t in ticks])
    ax.set_xlabel("Years since diagnosis")
    ax.set_xlim(0, max_months)


def _clip(t: np.ndarray, v: np.ndarray, max_months: float) -> tuple[np.ndarray, np.ndarray]:
    """Step curve cut exactly at max_months (the last value is carried to the edge)."""
    keep = t <= max_months
    t, v = t[keep], v[keep]
    return np.append(t, max_months), np.append(v, v[-1])


# --------------------------------------------------------------------------- audit
def plot_leakage_audit(summary: pd.DataFrame, path: Path, dpi: int, title: str | None = None) -> Path:
    s = summary.set_index("variant")
    rows = [
        ("original_features", "Original features\n(incl. overall_survival + months)"),
        ("without_overall_survival", "overall_survival removed"),
        ("without_survival_columns", "Both survival columns removed\n(leak-free)"),
    ]
    baseline = s.loc["majority_baseline", "accuracy_mean"]
    h = 3.2
    fig, ax = plt.subplots(figsize=(7.5, h))
    fig.subplots_adjust(left=0.33, right=0.95, top=_top(h), bottom=0.16)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.45, -0.95)
    ax.grid(axis="y", visible=False)
    for i, (key, _) in enumerate(rows):
        acc = s.loc[key, "accuracy_mean"]
        _rounded_hbar(ax, i, 0, acc, 0.42, CATEGORICAL[0])
        ax.text(acc + 0.012, i, f"{acc:.3f}", va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
    ax.plot([baseline, baseline], [len(rows) - 0.45, -0.55], color=INK_2, linewidth=1, zorder=4)
    ax.text(baseline, -0.62, f"Always predicting the majority class: {baseline:.3f}",
            fontsize=8, color=INK_2, va="bottom", ha="left")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl for _, lbl in rows])
    ax.set_xlabel("Accuracy (mean over repeated stratified 5-fold CV)")
    _heading(fig, title or "Accuracy of the original classifier with and without the survival columns",
             "Original 3-class logistic regression, re-run with all preprocessing inside the CV folds")
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- Kaplan-Meier
def plot_km(curves: dict[str, Any], at_risk: pd.DataFrame, colors: list[str], title: str, subtitle: str,
            path: Path, dpi: int, max_months: float = 240) -> Path:
    n = len(curves)
    h = 5.4
    fig = plt.figure(figsize=(7.5, h))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.3 * n + 0.45], hspace=0.3,
                          left=0.24, right=0.97, top=_top(h), bottom=0.03)
    ax = fig.add_subplot(gs[0])
    for (g, c), col in zip(curves.items(), colors):
        t, s = _clip(np.concatenate([[0], c["time"]]), np.concatenate([[1], c["surv"]]), max_months)
        ax.step(t, s, where="post", color=col, label=f"{g} (n={c['n']})")
    _years_axis(ax, max_months)
    ax.set_ylim(0, 1.02)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(v * 100))}%" for v in ticks])
    ax.set_ylabel("Free of breast-cancer death")
    ax.legend(loc="lower left", ncol=2 if n > 3 else 1, handlelength=1.6, columnspacing=1.4)

    # numbers at risk, aligned with the x ticks; short group names
    tab = fig.add_subplot(gs[1], sharex=ax)
    tab.set_ylim(n - 0.5, -1.4)
    tab.axis("off")
    yax = tab.get_yaxis_transform()
    tab.text(-0.26, -1.0, "Number at risk", transform=yax, ha="left", va="center",
             fontsize=8, color=INK_2, fontweight="bold")
    months = [int(col[:-1]) for col in at_risk.columns if col.endswith("m")]
    for i, ((g, _), col) in enumerate(zip(curves.items(), colors)):
        tab.add_line(Line2D([-0.26, -0.235], [i, i], transform=yax, color=col, linewidth=2.5,
                            solid_capstyle="round", clip_on=False))
        tab.text(-0.22, i, g.split(" (")[0], transform=yax, ha="left", va="center", fontsize=8, color=INK_2)
        row = at_risk.iloc[i]
        for m in months:
            if m <= max_months:
                tab.text(m, i, f"{row[f'{m}m']}", ha="center", va="center", fontsize=8, color=INK)
    _heading(fig, title, subtitle)
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- competing risks
def plot_cif(cif: dict[str, Any], path: Path, dpi: int, max_months: float = 240, title: str | None = None) -> Path:
    h = 3.7
    fig, axes = plt.subplots(1, 2, figsize=(7.5, h), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=_top(h, 0.25), bottom=0.15, wspace=0.08)
    panels = (("disease", "Death from breast cancer"), ("other", "Death from other causes"))
    ymax = 0
    for ax, (key, name) in zip(axes, panels):
        for (g, c), col in zip(cif.items(), ORDINAL_3):
            t, v = _clip(np.concatenate([[0], c["time"]]), np.concatenate([[0], c[key]]), max_months)
            ax.step(t, v, where="post", color=col, label=f"Age {g} (n={c['n']})")
            ymax = max(ymax, v.max())
        ax.set_title(name, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        _years_axis(ax, max_months)
    top = np.ceil(ymax * 10) / 10
    ticks = np.arange(0, top + 1e-9, 0.1)
    axes[0].set_ylim(0, top)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f"{int(round(v * 100))}%" for v in ticks])
    axes[0].set_ylabel("Cumulative incidence")
    axes[1].legend(loc="upper left", handlelength=1.6)
    _heading(fig, title or "Cumulative incidence of breast-cancer and other-cause death by age",
             "Aalen-Johansen estimator, the two causes of death treated as competing risks")
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- CV results
def plot_cindex_grid(summary: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int,
                     title: str | None = None) -> Path:
    endpoints = list(cfg["endpoints"])
    labels = {fs: spec["label"] for fs, spec in cfg["feature_sets"].items()}
    sets = list(cfg["feature_sets"])
    if (summary["feature_set"] == "age_only").any():      # reference row: age used directly as risk score
        sets = ["age_only"] + sets
        labels["age_only"] = "Age alone (reference)"
    h = 3.4
    fig, axes = plt.subplots(1, len(endpoints), figsize=(7.5, h), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.98, top=_top(h, 0.25), bottom=0.17, wspace=0.12)
    lo = (summary["cindex_mean"] - summary["cindex_std"]).min()
    hi = (summary["cindex_mean"] + summary["cindex_std"]).max()
    xlim = (np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20 + 0.02)
    for ax, ep in zip(axes, endpoints):
        sub = summary[summary["endpoint"] == ep].set_index("feature_set")
        for i, fs in enumerate(sets):
            m, sd = sub.loc[fs, "cindex_mean"], sub.loc[fs, "cindex_std"]
            col = MUTED if fs == "age_only" else CATEGORICAL[0]
            ax.plot([m - sd, m + sd], [i, i], color=col, linewidth=2, alpha=0.35)
            ax.plot(m, i, "o", color=col, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(m, i - 0.28, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_title(cfg["endpoints"][ep]["label"], loc="left", fontsize=9, color=INK, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(len(sets) - 0.5, -0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("C-index")
    axes[0].set_yticks(range(len(sets)))
    axes[0].set_yticklabels([labels[fs] for fs in sets])
    cv = cfg["cv"]
    _heading(fig, title or "Discrimination by feature set and endpoint",
             f"Elastic-net Cox, nested CV: mean and ±1 SD over {cv['n_splits'] * cv['n_repeats']} outer folds "
             f"({cv['n_splits']}-fold x {cv['n_repeats']} repeats)")
    return _save(fig, path, dpi)


PAIRED_ROWS = [(("clinical_subtypes", "clinical"), "Molecular subtypes added to clinical"),
               (("clinical_genomic", "clinical_subtypes"), "Genes added to clinical + subtypes")]


def plot_paired_deltas(deltas: pd.DataFrame, tests: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int,
                       title: str | None = None) -> Path:
    endpoints = list(cfg["endpoints"])
    h = 4.9
    fig, axes = plt.subplots(len(endpoints), 1, figsize=(7.5, h), sharex=True)
    fig.subplots_adjust(left=0.31, right=0.8, top=_top(h, 0.25), bottom=0.11, hspace=0.5)

    def pick(df, ep, comp):
        return df[(df["endpoint"] == ep) & (df["better"] == comp[0]) & (df["reference"] == comp[1])]

    comps = [c for c, _ in PAIRED_ROWS]
    sel_d = pd.concat([pick(deltas, ep, c) for ep in endpoints for c in comps])
    sel_t = pd.concat([pick(tests, ep, c) for ep in endpoints for c in comps])
    span = max(sel_d["delta"].abs().max(), sel_t[["ci_low", "ci_high"]].abs().to_numpy().max()) * 1.12
    rng = np.random.default_rng(0)
    for ax, ep in zip(axes, endpoints):
        for i, (comp, _) in enumerate(PAIRED_ROWS):
            d = pick(deltas, ep, comp)
            ax.scatter(d["delta"], i + rng.uniform(-0.13, 0.13, len(d)), s=16, color=CATEGORICAL[0],
                       alpha=0.6, edgecolor=SURFACE, linewidth=0.8, zorder=3)
            t = pick(tests, ep, comp).iloc[0]
            ax.plot([t["ci_low"], t["ci_high"]], [i, i], color=INK, linewidth=1.6, zorder=4, solid_capstyle="butt")
            ax.plot(t["mean_delta"], i, "D", color=INK, markersize=5, zorder=5)
            p = t["p_value"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(1.02, i, f"{_signed(t['mean_delta'])}  ({ptxt})", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8, color=INK)
        ax.axvline(0, color=INK_2, linewidth=1, zorder=2)
        ax.set_ylim(len(PAIRED_ROWS) - 0.5, -0.6)
        ax.set_yticks(range(len(PAIRED_ROWS)))
        ax.set_yticklabels([lbl for _, lbl in PAIRED_ROWS], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_title(cfg["endpoints"][ep]["label"], loc="left", fontsize=9, color=INK, fontweight="bold")
        ax.set_xlim(-span, span)
    axes[-1].set_xlabel("Change in C-index per outer fold (dots); mean and corrected 95% CI (black)")
    _heading(fig, title or "Paired change in C-index as data types are added",
             "Same outer folds for every model; Nadeau-Bengio corrected resampled t-test")
    return _save(fig, path, dpi)


def plot_model_comparison(summary: pd.DataFrame, cfg: dict[str, Any], path: Path, dpi: int,
                          title: str | None = None) -> Path:
    mc = cfg["model_comparison"]
    models, sets = mc["models"], mc["feature_sets"]
    h = 3.1
    fig, ax = plt.subplots(figsize=(7.5, h))
    fig.subplots_adjust(left=0.27, right=0.97, top=_top(h, 0.3), bottom=0.18)
    lo = (summary["cindex_mean"] - summary["cindex_std"]).min()
    hi = (summary["cindex_mean"] + summary["cindex_std"]).max()
    offsets = np.linspace(-0.14, 0.14, len(sets))
    for (fs, off), col in zip(zip(sets, offsets), CATEGORICAL):
        for i, m in enumerate(models):
            r = summary[(summary["model"] == m) & (summary["feature_set"] == fs)].iloc[0]
            ax.plot([r["cindex_mean"] - r["cindex_std"], r["cindex_mean"] + r["cindex_std"]], [i + off] * 2,
                    color=col, linewidth=2, alpha=0.35)
            ax.plot(r["cindex_mean"], i + off, "o", color=col, markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=cfg["feature_sets"][fs]["label"] if i == 0 else None)
            ax.text(r["cindex_mean"] + r["cindex_std"] + 0.002, i + off, f"{r['cindex_mean']:.3f}",
                    va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlim(np.floor(lo * 50) / 50, np.ceil(hi * 50) / 50 + 0.015)
    ax.grid(axis="y", visible=False)
    n_folds = cfg["cv"]["n_splits"] * mc["n_repeats"]
    ax.set_xlabel(f"C-index (mean ± 1 SD over {n_folds} outer folds)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(sets), handletextpad=0.3)
    _heading(fig, title or "Model families compared on the same folds",
             f"{cfg['endpoints'][mc['endpoint']]['label']}; identical outer folds for every model")
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- explanations
def plot_shap_beeswarm(explanation, top_n: int, path: Path, dpi: int) -> Path:
    import shap
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_ramp", SEQUENTIAL)
    h = 6.0
    shap.plots.beeswarm(explanation, max_display=top_n + 1, color=cmap, show=False, plot_size=(7.5, h))
    fig = plt.gcf()
    ax, cax = fig.axes[0], fig.axes[-1]
    fig.subplots_adjust(left=0.36, right=0.9, top=_top(h), bottom=0.1)
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="y", labelsize=8.5, labelcolor=INK_2)
    ax.tick_params(axis="x", labelsize=8.5, labelcolor=INK_2)
    ax.set_xlabel("SHAP value (contribution to the log hazard of breast-cancer death)", fontsize=9, color=INK_2)
    cax.tick_params(labelsize=8, labelcolor=INK_2)
    cax.set_ylabel("Feature value", fontsize=8.5, color=INK_2)
    _heading(fig, "What the gradient-boosted Cox model relies on",
             "TreeSHAP for every patient (one dot each); darker dots = higher feature value")
    return _save(fig, path, dpi)


def plot_shap_groups(groups: pd.DataFrame, path: Path, dpi: int, title: str | None = None) -> Path:
    h = 2.8
    fig, ax = plt.subplots(figsize=(7.5, h))
    fig.subplots_adjust(left=0.3, right=0.93, top=_top(h), bottom=0.2)
    g = groups.reset_index(drop=True)
    ax.set_xlim(0, max(0.6, g["share"].max() * 1.2))
    ax.set_ylim(len(g) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for i, r in g.iterrows():
        _rounded_hbar(ax, i, 0, r["share"], 0.42, CATEGORICAL[0])
        ax.text(r["share"] + 0.01, i, f"{r['share'] * 100:.0f}%", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{r['group']} ({r['n_features']} features)" for _, r in g.iterrows()])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("Share of the summed mean |SHAP| across all features")
    _heading(fig, title or "Share of SHAP attribution by data type",
             "Gradient-boosted Cox model for breast-cancer death, all feature groups available")
    return _save(fig, path, dpi)


def plot_coefficients(coefs: pd.DataFrame, top_n: int, path: Path, dpi: int) -> Path:
    c = coefs.head(top_n).reset_index(drop=True)
    h = 0.27 * len(c) + 1.9
    fig, ax = plt.subplots(figsize=(7.5, h))
    fig.subplots_adjust(left=0.38, right=0.8, top=_top(h, 0.3), bottom=0.6 / h)
    span = c["coef"].abs().max() * 1.12
    ax.set_xlim(-span, span)
    ax.set_ylim(len(c) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    has_freq = "selection_frequency" in c
    for i, r in c.iterrows():
        _rounded_hbar(ax, i, 0, r["coef"], 0.55, HARMFUL if r["coef"] > 0 else PROTECTIVE)
        if has_freq:
            ax.text(1.03, i, f"{r['selection_frequency'] * 100:.0f}%", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8, color=INK)
    if has_freq:
        ax.text(1.03, -0.7, "Selected in\nCV folds", transform=ax.get_yaxis_transform(), va="bottom",
                ha="left", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c["label"], fontsize=8)
    ax.set_xlabel("Coefficient (log hazard ratio per SD of the standardized feature)")
    handles = [Line2D([], [], color=HARMFUL, linewidth=6, label="Higher risk"),
               Line2D([], [], color=PROTECTIVE, linewidth=6, label="Lower risk")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.0)
    _heading(fig, f"Top {len(c)} elastic-net Cox coefficients for breast-cancer death",
             "Model refit on all patients (penalty chosen by CV); right: share of outer CV folds selecting it")
    return _save(fig, path, dpi)
