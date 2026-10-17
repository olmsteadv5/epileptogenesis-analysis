
from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .shared import (
    DURING_COLOR,
    MEAN_COLOR,
    R2_CMAP,
    RACINE_CMAP,
    RACINE_SCALE_MAX,
    _ANIMAL_MARKERS,
    _DEFAULT_DAY_ZERO,
    Panel,
    _animal_colors,
    _band_label,
    _bin_size_legend,
    _bins_to_size,
    _duration_size_legend,
    _duration_to_size,
    longitudinal_animal_figure,
    longitudinal_global_figure,
)


def plot_phase_delay_seizure(
    coh,
    band_low=(0.0, 14.0),
    title: str | None = None,
):
    # explainer: each point is a coherent bin coloured by MSC, MSC-weighted fit through them, delay/R2/bins in the corner
    lo, hi = band_low
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")

    # explainer: (freqs, phase, unused-freqs, unused-phase, slope, intercept, msc-freqs, msc-curve, tau, r2, nbins, label)
    panels = [
        ("phase_pre_freqs", "phase_pre_values", "phase_pre_unused_freqs", "phase_pre_unused_values",
         "phase_pre_slope", "phase_pre_intercept", "freqs_pre", "msc_pre",
         "tau_ms_pre", "phase_r2_pre", "phase_nbins_pre", "Pre-Seizure"),
        ("phase_sz_freqs", "phase_sz_values", "phase_sz_unused_freqs", "phase_sz_unused_values",
         "phase_sz_slope", "phase_sz_intercept", "freqs_seizure", "msc_seizure",
         "tau_ms_seizure", "phase_r2_seizure", "phase_nbins_seizure", "During Seizure"),
    ]

    mappable = None
    for ax, (fk, vk, ufk, uvk, sk, ik, fmk, vmk, tauk, r2k, nbk, label) in zip(axes, panels):
        # decision: grey discarded bins first so they sit behind the used ones
        uf = np.asarray(coh[ufk], dtype=float)
        uv = np.asarray(coh[uvk], dtype=float)
        if uf.size >= 1:
            ax.scatter(uf, uv, color="lightgray", s=14, edgecolors="none", zorder=1)

        pf = np.asarray(coh[fk], dtype=float)
        pv = np.asarray(coh[vk], dtype=float)

        if pf.size >= 1:
            msc_pts = np.interp(pf, np.asarray(coh[fmk]), np.asarray(coh[vmk]))
            mappable = ax.scatter(pf, pv, c=msc_pts, cmap=R2_CMAP, vmin=0.0, vmax=1.0,
                                  s=22, edgecolors="black", linewidths=0.3, zorder=3)

            slope, intercept = coh[sk], coh[ik]
            if np.isfinite(slope):
                xline = np.array([pf.min(), pf.max()])
                ax.plot(xline, slope * xline + intercept, color=DURING_COLOR,
                        linewidth=1.5, zorder=2)

        ax.text(0.03, 0.97,
                f"τ = {coh[tauk]:.2f} ms\nR² = {coh[r2k]:.3f}\nbins = {coh[nbk]}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
        ax.set_title(label)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Frequency (Hz)")

    axes[0].set_ylabel("Unwrapped phase (rad)")

    grey_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="lightgray",
                             markeredgecolor="none", markersize=6,
                             label="Below MSC threshold (not used in fit)")
    axes[0].legend(handles=[grey_handle], loc="lower left", fontsize=8, framealpha=0.85)

    if mappable is not None:
        fig.colorbar(mappable, ax=list(axes), fraction=0.025, pad=0.02).set_label("MSC")
    if title:
        fig.suptitle(title)
    return fig


def plot_phase_by_animal(
    coherence_details,
    band_low=(0.0, 14.0),
    which: str = "seizure",
    align_phase: bool = True,
    phase_ylim=None,
    color_by: str = "msc",  # explainer: msc = opacity, colorbar = viridis + colorbar
    alpha_floor: float = 0.08,
):
    if which == "seizure":
        f_ph, v_ph = "phase_sz_freqs", "phase_sz_values"
        f_msc, v_msc = "freqs_seizure", "msc_seizure"
    else:
        f_ph, v_ph = "phase_pre_freqs", "phase_pre_values"
        f_msc, v_msc = "freqs_pre", "msc_pre"

    lo, hi = band_low
    animals = sorted({d["Animal_ID"] for d in coherence_details})
    colors = _animal_colors(animals)

    n = len(animals)
    fig, raw_axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.2), sharex=True, sharey=True, squeeze=False,
        layout="constrained" if color_by == "colorbar" else None,
    )
    axes = raw_axes[0]
    mappable = None

    for ax, animal in zip(axes, animals):
        base = mcolors.to_rgb(colors[animal])
        for d in coherence_details:
            if d["Animal_ID"] != animal:
                continue
            pf = np.asarray(d[f_ph], dtype=float)
            pv = np.asarray(d[v_ph], dtype=float)
            if pf.size < 2:
                continue
            if align_phase:
                pv = pv - pv[0]
            msc_pts = np.interp(pf, np.asarray(d[f_msc]), np.asarray(d[v_msc]))

            if color_by == "msc":
                a = np.clip(msc_pts, alpha_floor, 1.0)
                rgba = np.zeros((len(a), 4)); rgba[:, :3] = base; rgba[:, 3] = a
                ax.plot(pf, pv, color=base, alpha=0.12, linewidth=0.8)
                ax.scatter(pf, pv, c=rgba, s=12, edgecolors="none")
            else:
                ax.plot(pf, pv, color="gray", alpha=0.15, linewidth=0.8)
                mappable = ax.scatter(pf, pv, c=msc_pts, cmap="viridis",
                                      vmin=0.0, vmax=1.0, s=12, edgecolors="none")

        ax.set_title(f"Animal {animal}")
        ax.set_xlim(lo, hi)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        if phase_ylim is not None:
            ax.set_ylim(*phase_ylim)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Phase (rad)" + (", aligned to 0" if align_phase else ""))

    if color_by == "colorbar" and mappable is not None:
        fig.colorbar(mappable, ax=list(axes), fraction=0.025, pad=0.02).set_label("MSC")
    fig.suptitle(f"Phase responses by animal ({_band_label(band_low)}, {which})")
    if color_by != "colorbar":
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_delay_distribution(
    df: pd.DataFrame,
    group_col: str = "Animal_ID",
    value_col: str = "Tau_ms_Seizure",
    r2_col: str = "Phase_R2_Seizure",
    nbins_col: str = "Phase_Nbins_Seizure",
    r2_min: float = 0.0,
    nbins_min: int = 3,
    title: str | None = None,
):
    # decision: only fits that cleared r2_min and nbins_min, so the spread reflects delays we trust
    d = df[df[value_col].notna()].copy()
    if r2_col in d:
        d = d[d[r2_col].fillna(0) >= r2_min]
    if nbins_col in d:
        d = d[d[nbins_col].fillna(0) >= nbins_min]

    groups = sorted(d[group_col].unique())
    data = [d.loc[d[group_col] == g, value_col].values for g in groups]
    colors = _animal_colors(groups)

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.55,
                    showfliers=False, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[g]); patch.set_alpha(0.35)
    for i, (g, arr) in enumerate(zip(groups, data)):
        jitter = (np.random.default_rng(i).random(len(arr)) - 0.5) * 0.25
        ax.scatter(np.full(len(arr), i) + jitter, arr, color=colors[g],
                   s=18, alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"Animal {g}\n(n={len(a)})" for g, a in zip(groups, data)])
    ax.set_ylabel("Phase delay (ms)")
    ax.set_title(title or f"Per-seizure delay by animal "
                          f"(r2>={r2_min}, bins>={nbins_min})")
    fig.tight_layout()
    return fig


def _draw_confidence(tau_col, r2_col, nbin_col, cmap):
    def draw(ax, x, g):
        sc = ax.scatter(
            x, g[tau_col], c=g[r2_col], cmap=cmap, vmin=0.0, vmax=1.0,
            s=_bins_to_size(g[nbin_col]), alpha=0.9, edgecolors="black",
            linewidths=0.4, zorder=3,
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        return sc
    return draw


# explainer: confidence folds R2, bin count and MSC into one 0-1 score, see compute_phase_delay
def _filter_phase_points(df, conf_min):
    return df[df["Phase_Confidence_Seizure"] >= conf_min]


def plot_phase_delay_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    pre_seizure_secs: float = 30,
    cmap: str = R2_CMAP,
):
    animal = group_df["Animal_ID"].iloc[0]
    bin_legend = lambda ax, sc: _bin_size_legend(ax, sc, loc="upper right")

    panels = [
        Panel(f"During Seizure ({_band_label(band_low)}) | Animal: {animal}",
              "Time Delay (ms)",
              _draw_confidence("Tau_ms_Seizure", "Phase_R2_Seizure",
                               "Phase_Nbins_Seizure", cmap),
              after=bin_legend),
        Panel(f"Pre-Seizure {pre_seizure_secs:g}s ({_band_label(band_low)}) | Animal: {animal}",
              "Time Delay (ms)",
              _draw_confidence("Tau_ms_Pre", "Phase_R2_Pre", "Phase_Nbins_Pre", cmap),
              after=bin_legend),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(11, 10), sharex=True,
        colorbar_label="Phase-slope fit R²",
        suptitle=f"Longitudinal Phase Delay | Animal {animal}",
    )


def plot_phase_delay_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
    cmap: str = R2_CMAP,
    conf_min: float = 0.0,
    ylim: tuple[float, float] | None = None,
    xlim: tuple | None = None,
):
    # decision: marker shape = animal so colour stays free for fit quality. ylim/xlim lock the axes across sweep frames
    normalized_df = _filter_phase_points(normalized_df, conf_min)
    animals = list(dict.fromkeys(normalized_df["Animal_ID"]))
    marker_for = {a: _ANIMAL_MARKERS[i % len(_ANIMAL_MARKERS)] for i, a in enumerate(animals)}

    def draw(ax, df, _colors):
        sc = None
        for a in animals:
            g = df[df["Animal_ID"] == a]
            sc = ax.scatter(
                g["Normalized_DT"], g["Tau_ms_Seizure"], c=g["Phase_R2_Seizure"],
                cmap=cmap, vmin=0.0, vmax=1.0, s=_bins_to_size(g["Phase_Nbins_Seizure"]),
                marker=marker_for[a], alpha=0.9, edgecolors="black", linewidths=0.4,
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        return sc

    def after(ax, sc):
        if sc is None:
            return
        shape_handles = [
            plt.Line2D([0], [0], marker=marker_for[a], linestyle="none",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=8, label=str(a))
            for a in animals
        ]
        animal_leg = ax.legend(handles=shape_handles, title="Animal", loc="upper left",
                               framealpha=0.9)
        ax.add_artist(animal_leg)
        _bin_size_legend(ax, sc, loc="upper right")

    panels = [
        Panel(f"Global Phase Time Delay ({_band_label(band_low)})", "Time Delay (ms)",
              draw, ylim=ylim, after=after),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(12, 7),
        colorbar_label="Phase-slope fit R²", xlim=xlim,
        suptitle="Population Phase Delay: Offset to Experimental Day 0",
    )


# decision: plotly symbols matched to the matplotlib markers so animals look the same in the PNG and the HTML
_PLOTLY_MARKERS = ["circle", "square", "triangle-up", "diamond", "triangle-down",
                   "cross", "x", "star", "triangle-left", "triangle-right"]


def plot_phase_delay_global_interactive(
    normalized_df: pd.DataFrame,
    html_path: str,
    band_low=(0.0, 14.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
    conf_min: float = 0.0,
    conf_step: float = 0.1,
    ylim: tuple[float, float] | None = None,
    xlim: tuple | None = None,
    path_cols=("Phase_Delay_Plot", "Coherence_Plot", "Correlation_Plot"),
):
    # explainer: interactive twin of plot_phase_delay_global_summary. click a point to open its per-seizure PNGs
    import os
    from datetime import datetime

    import numpy as np
    import plotly.graph_objects as go

    df = normalized_df.copy()
    animals = list(dict.fromkeys(df["Animal_ID"]))

    epoch = pd.Timestamp(datetime(*day_zero))
    df["_Day"] = (df["Normalized_DT"] - epoch).dt.total_seconds() / 86400.0

    # decision: links relative to where the HTML lands so it works from file://
    html_dir = os.path.dirname(os.path.abspath(html_path))
    def _rel(p):
        if isinstance(p, str) and p:
            return os.path.relpath(os.path.abspath(p), start=html_dir)
        return ""
    have_paths = all(c in df.columns for c in path_cols)

    def _stats(g):
        rac = ("<br>Racine = " + g["Racine_Scale"].astype(str)) if "Racine_Scale" in g else ""
        # decision: real seizure time first, then file start, then the normalized timeline
        when_col = next((c for c in ("Seizure_DT", "File_Start_DT", "Normalized_DT") if c in g),
                        "Normalized_DT")
        when = pd.to_datetime(g[when_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return (
            "Animal " + g["Animal_ID"].astype(str)
            + "<br>" + when
            + "<br>τ = " + g["Tau_ms_Seizure"].round(2).astype(str) + " ms"
            + "<br>R² = " + g["Phase_R2_Seizure"].round(3).astype(str)
            + "<br>bins = " + g["Phase_Nbins_Seizure"].astype(str)
            + rac
        ).tolist()

    def _custom(g):
        # explainer: customdata = [stats_html, phase_delay_url, coherence_url, correlation_url], the hover card turns the urls into links
        stats = _stats(g)
        if have_paths:
            links = [[_rel(r[c]) for c in path_cols] for _, r in g.iterrows()]
        else:
            links = [["", "", ""] for _ in range(len(g))]
        return [[stats[i]] + links[i] for i in range(len(g))]

    # explainer: per-animal arrays at one cutoff, used for the first draw and every slider restyle
    def arrays_for(conf):
        sub = df[df["Phase_Confidence_Seizure"] >= conf]
        xs, ys, sizes, colors, customs = [], [], [], [], []
        for a in animals:
            g = sub[sub["Animal_ID"] == a]
            xs.append(g["_Day"].tolist())
            ys.append(g["Tau_ms_Seizure"].tolist())
            sizes.append((6 + g["Phase_Nbins_Seizure"].clip(0, 20)).tolist())
            colors.append(g["Phase_R2_Seizure"].tolist())
            customs.append(_custom(g))
        return xs, ys, sizes, colors, customs

    x0, y0, s0, c0, cd0 = arrays_for(conf_min)
    fig = go.Figure()
    for i, a in enumerate(animals):
        fig.add_trace(go.Scatter(
            x=x0[i], y=y0[i], mode="markers", name=str(a),
            marker=dict(
                symbol=_PLOTLY_MARKERS[i % len(_PLOTLY_MARKERS)],
                size=s0[i], color=c0[i], coloraxis="coloraxis", opacity=0.85,
                line=dict(width=0.6, color="black"),
            ),
            # decision: native hover off, the custom card shows stats + links
            customdata=cd0[i], hoverinfo="none",
        ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=1)

    # explainer: each slider step restyles every trace with its arrays at that cutoff
    steps = np.round(np.arange(0.0, 1.0 + 1e-9, conf_step), 2)
    slider_steps = []
    for s in steps:
        xs, ys, sizes, colors, customs = arrays_for(s)
        slider_steps.append(dict(
            method="restyle", label=f"{s:.2f}",
            args=[{"x": xs, "y": ys, "marker.size": sizes, "marker.color": colors,
                   "customdata": customs}],
        ))
    active = int(np.argmin(np.abs(steps - conf_min)))

    fig.update_layout(
        title=f"Population Phase Delay ({_band_label(band_low)}): Offset to Experimental Day 0",
        xaxis_title="Experimental Days (Starting from Day 0)",
        yaxis_title="Time Delay (ms)",
        # decision: legend above the colorbar so they don't overlap
        legend=dict(title="Animal", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        coloraxis=dict(colorscale=R2_CMAP, cmin=0.0, cmax=1.0,
                       colorbar=dict(title="Phase-slope fit R²", x=1.02, xanchor="left",
                                     y=0.45, yanchor="top", len=0.55)),
        sliders=[dict(active=active, currentvalue=dict(prefix="Min phase-fit confidence: "),
                      pad=dict(t=40), steps=slider_steps)],
        template="simple_white", width=1150, height=680,
        margin=dict(r=170),
    )
    # decision: fixed x range so filtering never rescales it, zoom and pan still work
    if xlim is None:
        dmin, dmax = float(df["_Day"].min()), float(df["_Day"].max())
        pad = 0.02 * (dmax - dmin) if dmax > dmin else 1.0
        xlim = (dmin - pad, dmax + pad)
    fig.update_xaxes(range=list(xlim), autorange=False)
    if ylim is not None:
        fig.update_yaxes(range=list(ylim))

    # decision: plotly tooltips vanish when the mouse leaves the point, so links can't be clicked. own card lingers briefly and stays while hovered
    post_script = """
    var gd = document.getElementById('{plot_id}');

    var card = document.createElement('div');
    card.style.cssText = 'position:absolute;z-index:1000;display:none;background:#fff;'
        + 'border:1px solid #888;border-radius:6px;padding:8px 10px;font:12px sans-serif;'
        + 'box-shadow:0 2px 6px rgba(0,0,0,0.25);max-width:280px;line-height:1.5;';
    document.body.appendChild(card);

    var hideTimer = null;
    function hideCard() { card.style.display = 'none'; }
    card.addEventListener('mouseenter', function() { if (hideTimer) clearTimeout(hideTimer); });
    card.addEventListener('mouseleave', hideCard);

    var LABELS = ['Phase-frequency regression', 'Coherence (pre vs during)', 'Correlation'];

    gd.on('plotly_hover', function(data) {
        if (hideTimer) clearTimeout(hideTimer);
        var cd = data.points[0].customdata || [];
        var html = cd[0] || '';
        var links = '';
        for (var i = 0; i < LABELS.length; i++) {
            var u = cd[i + 1];
            if (u) {
                links += '<a href="' + u + '" target="_blank" rel="noopener">'
                       + LABELS[i] + '</a><br>';
            }
        }
        if (links) { html += '<hr style="margin:6px 0">' + links; }
        card.innerHTML = html;
        var e = data.event;
        card.style.left = (e.pageX + 12) + 'px';
        card.style.top = (e.pageY + 12) + 'px';
        card.style.display = 'block';
    });

    gd.on('plotly_unhover', function() { hideTimer = setTimeout(hideCard, 400); });
    """

    os.makedirs(html_dir, exist_ok=True)
    # decision: wheel zoom on top of box zoom to pull clusters apart
    fig.write_html(html_path, post_script=post_script, include_plotlyjs="cdn",
                   config={"scrollZoom": True, "displaylogo": False})
    return fig


def plot_phase_delay_during_animal_summary(
    group_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    racine_cmap: str = RACINE_CMAP,
    racine_max: int = RACINE_SCALE_MAX,
):
    # explainer: colour = Racine severity, size = seizure duration, dashed line = mean delay
    animal = group_df["Animal_ID"].iloc[0]

    def draw(ax, x, g):
        ax.plot(x, g["Tau_ms_Seizure"], color="gray", linewidth=1.5, alpha=0.4, label="Trend")
        sc = ax.scatter(
            x, g["Tau_ms_Seizure"], c=g["Racine_Scale"], cmap=racine_cmap,
            vmin=1, vmax=racine_max, s=_duration_to_size(g["Seizure_Duration_Sec"]),
            alpha=0.9, edgecolors="black", linewidths=0.4, zorder=3,
        )
        # decision: mean skips seizures whose phase fit failed (NaN tau)
        mean_val = float(np.nanmean(g["Tau_ms_Seizure"]))
        ax.axhline(mean_val, color=MEAN_COLOR, linestyle="--", linewidth=2,
                   label=f"Mean: {round(mean_val, 2)} ms")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.legend(loc="upper left")
        return sc

    dur_legend = lambda ax, sc: _duration_size_legend(ax, sc, loc="upper right")
    panels = [
        Panel(f"During Seizure Phase Delay ({_band_label(band_low)}) | Animal: {animal}",
              "Time Delay (ms)", draw, after=dur_legend),
    ]
    return longitudinal_animal_figure(
        group_df, panels, figsize=(11, 6),
        colorbar_label="Racine severity",
        suptitle=f"Longitudinal During-Seizure Phase Delay | Animal {animal}",
    )


def plot_phase_delay_during_global_summary(
    normalized_df: pd.DataFrame,
    band_low=(0.0, 14.0),
    day_zero: tuple[int, int, int] = _DEFAULT_DAY_ZERO,
    racine_cmap: str = RACINE_CMAP,
    racine_max: int = RACINE_SCALE_MAX,
    conf_min: float = 0.0,
):
    # explainer: colour = Racine severity, size = duration, marker = animal. conf_min drops low-confidence fits
    normalized_df = _filter_phase_points(normalized_df, conf_min)
    animals = list(dict.fromkeys(normalized_df["Animal_ID"]))
    marker_for = {a: _ANIMAL_MARKERS[i % len(_ANIMAL_MARKERS)] for i, a in enumerate(animals)}

    def draw(ax, df, _colors):
        sc = None
        for a in animals:
            g = df[df["Animal_ID"] == a]
            sc = ax.scatter(
                g["Normalized_DT"], g["Tau_ms_Seizure"], c=g["Racine_Scale"],
                cmap=racine_cmap, vmin=1, vmax=racine_max,
                s=_duration_to_size(g["Seizure_Duration_Sec"]),
                marker=marker_for[a], alpha=0.9, edgecolors="black", linewidths=0.4,
            )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        return sc

    def after(ax, sc):
        if sc is None:
            return
        shape_handles = [
            plt.Line2D([0], [0], marker=marker_for[a], linestyle="none",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=8, label=str(a))
            for a in animals
        ]
        animal_leg = ax.legend(handles=shape_handles, title="Animal", loc="upper left",
                               framealpha=0.9)
        ax.add_artist(animal_leg)
        _duration_size_legend(ax, sc, loc="upper right")

    panels = [
        Panel(f"Global During-Seizure Phase Delay ({_band_label(band_low)})",
              "Time Delay (ms)", draw, after=after),
    ]
    return longitudinal_global_figure(
        normalized_df, panels, day_zero=day_zero, figsize=(12, 7),
        colorbar_label="Racine severity",
        suptitle="Population During-Seizure Phase Delay: Offset to Experimental Day 0",
    )
