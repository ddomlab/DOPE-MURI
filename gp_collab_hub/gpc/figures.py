"""Every figure the report saves, and every setting that controls them.

THIS IS THE FILE TO EDIT. Titles, axis labels, colours, figure sizes, bar
widths, tick angles, label positions, axis limits -- all of it lives in `STYLE`
below, and `gpc/report.py` only calls the functions here. Nothing else in the
package draws anything that gets written to disk, so a change made here shows up
in every dataset's results folder the next time the report is regenerated.

The figures:

    pooled_by_featureset   pooled out-of-fold metric per feature section, one
                           group of bars per evaluation method
    lolo_by_group          the held-out-group summary: one bar per held-out
                           ligand/catalyst, one colour per feature section
    kraken_space           two panels -- PC1 vs PC2, and the two buried-volume
                           descriptors -- with the full Kraken population behind
                           this dataset's own ligands

`STYLE` is a plain nested dict, so a notebook can poke at one value
(`figures.STYLE["pooled"]["figsize"] = (11, 5)`) for a one-off without editing
the file. Edit the file when you want the change to stick.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# STYLE -- the whole knob board.
# ---------------------------------------------------------------------------
STYLE = {
    # Applies to every figure.
    "common": {
        "dpi": 300,
        "facecolor": "white",
        "spines_off": ("top", "right"),
        "grid_alpha": 0.25,
        "grid_linewidth": 0.6,
        "bar_label_fontsize": 7,
        "bar_label_padding": 2,
        "bar_label_format": "%.2f",
        # The y axis floor is pinned at 0, so a negative score draws below the
        # axis and would otherwise just be a missing bar. These settings label it
        # at the base instead: the value stays visible and the baseline stays at
        # zero, where it belongs.
        "clipped_label_color": "#b03a2e",
        "clipped_label_fontsize": 6.5,
    },

    # Pooled metric by feature section. The title keeps the word "Pooled"; the
    # y axis deliberately does not, because the axis is just the metric.
    "pooled": {
        "figsize": (9, 4.5),
        "title": "{dataset}: Pooled {metric} by feature selection",
        "title_fontsize": 11,
        "ylabel": "{metric}{unit}",
        "xlabel": "",
        "bar_width": 0.78,
        "bar_edgecolor": "white",
        "bar_edgewidth": 0.6,
        "tick_rotation": 15,
        # One colour per evaluation method, matching the per-project reviews.
        "method_colors": {"lolo": "#c0392b", "iid_matched": "#5b8db8",
                          "kfold": "#a8c8e0", "holdout": "#7f8c8d"},
        "fallback_colors": ["#c0392b", "#5b8db8", "#a8c8e0", "#7f8c8d", "#d9a441"],
        "method_order": ("lolo", "iid_matched", "kfold"),
        "legend_title": "Evaluation method",
        "legend_fontsize": 8,
        "legend_loc": "upper left",
        "legend_bbox": (1.01, 1.0),
        # Headroom above the tallest bar. The floor is pinned at 0 so a negative
        # R^2 reads against zero rather than against a shifted baseline; the
        # bar still draws downward and its label sits at the axis.
        "ylim_headroom": 1.12,
        "ylim_bottom": 0.0,
        "clipped_label_rotation": 0,
    },

    # The held-out-group summary.
    "lolo": {
        "figsize": (10, 4.2),
        "title": "{dataset}: LOLO {metric} by held-out {group}",
        "title_fontsize": 11,
        "xlabel": "Held-out {group}",
        "ylabel": "{metric}{unit}",
        "bar_width": 0.85,
        "tick_rotation": 35,
        # Okabe-Ito, colour-blind safe, one per feature section.
        "featureset_colors": ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                              "#56B4E9", "#D55E00", "#F0E442", "#000000"],
        "legend_title": "Feature set",
        "legend_fontsize": 8,
        "legend_loc": "upper left",
        "legend_bbox": (1.01, 1.0),
        "bar_labels": False,          # 5 sections x 8 ligands is too many to label
        "ylim_headroom": 1.12,
        "ylim_bottom": 0.0,
        "clipped_label_rotation": 90,   # the bars are narrow here
    },

    # PC1/PC2 and the buried-volume pair, side by side.
    "kraken": {
        "figsize": (13.5, 6.0),
        "suptitle": "{dataset}: ligands in the Kraken feature space",
        "suptitle_fontsize": 15,
        "panel_titles": ("Kraken Database", "Buried volume"),
        "panel_title_fontsize": 13,
        "axis_label_fontsize": 12,
        "tick_labelsize": 11,
        "spine_linewidth": 1.8,
        "spine_color": "0.15",
        "tick_width": 1.8,
        "tick_length": 6,

        # The reference population.
        "population_color": "0.72",
        "population_size": 18,
        "population_alpha": 1.0,
        "population_label": "Kraken reference ({n})",

        # This dataset's own ligands.
        "highlight_color": "#0d3b66",
        "highlight_size": 95,
        "highlight_edgecolor": "white",
        "highlight_edgewidth": 0.8,
        "highlight_label": "{dataset} ({n})",

        # Point labels. `offsets` overrides the automatic placement for a named
        # ligand: {"XPhos": (18, -14)} in points. Everything not named is placed
        # by pushing away from its nearest neighbours, so a new dataset draws
        # legibly without anyone tuning it first.
        "labels": True,
        "label_fontsize": 10,
        "label_weight": "bold",
        "label_radius": 24,              # points, how far a label sits from its dot
        "label_box_alpha": 0.78,
        "leader_color": "0.35",
        "leader_linewidth": 1.0,
        "leader_shrink": 6,
        "offsets": {},

        "legend_fontsize": 10,
        "legend_loc": "upper left",

        # Panel A axes. The "(x% of descriptor variance)" suffix is deliberately
        # absent: these are the frozen reference components, and quoting the
        # variance on every figure invites reading one dataset's share against
        # another's when the PCA is the same one either way.
        "pc_x": "PC1",
        "pc_y": "PC2",
        "pc_xlabel": "PC1",
        "pc_ylabel": "PC2",
        "pc_xlim": None,                 # None -> matplotlib picks
        "pc_ylim": (-20, 30),
        "zero_lines": False,             # the cross-hairs at x=0 / y=0

        # Panel B axes.
        "vbur_x": "vbur_pct_boltz",
        "vbur_y": "vbur_pct_min",
        "vbur_xlabel": r"$V_\mathrm{bur}$, Boltzmann average (%)",
        "vbur_ylabel": r"$V_\mathrm{bur}$, minimum (%)",
        "vbur_xlim": None,
        "vbur_ylim": None,
    },
}

# How a metric key is written on an axis, and what unit it carries. RMSE and MAE
# are in the target's units; R^2 and Kendall tau are dimensionless.
METRIC_LABELS = {"r2": "R$^2$", "rmse": "RMSE", "mae": "MAE",
                 "kendall_tau": r"Kendall $\tau$", "predictive_nll": "NLL"}
METRIC_UNITS = {"rmse": " (% yield)", "mae": " (% yield)"}

# Display names for the saved figures. `gpc/results.py` has its own map, which
# keeps the raw config keys for interactive work; these are the presentation
# versions, and the three spellings of the one-hot section collapse to one label
# so a figure reads the same whichever dataset produced it.
FEATURESET_LABELS = {
    "group_ohe": "OHE", "ligand_ohe": "OHE", "catalyst_ohe": "OHE",
    "selected_2": "Selected-2", "selected_5": "Selected-5",
    # The two steric+electronic pairs. Spelled out rather than numbered,
    # because which buried volume each carries is the whole point of them.
    "vbur_min_vmin": r"$V_\mathrm{bur}$ min + $V_\mathrm{min}$",
    "vbur_boltz_vmin": r"$V_\mathrm{bur}$ boltz + $V_\mathrm{min}$",
    "pc_top": "PC top", "pc_scores": "PC scores",
    "pc_scores_long": "PC scores (long)",
    "rxnpredict_full": "Published DFT", "rxnpredict_full_ohe": "Published DFT + OHE",
}
METHOD_DISPLAY = {"lolo": "LOLO", "iid_matched": "Matched IID", "kfold": "5-fold CV",
                  "kfold_5": "5-fold CV (unstratified)", "holdout": "80:20 split"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.upper())


def metric_unit(metric: str) -> str:
    return METRIC_UNITS.get(metric, "")


def featureset_label(model: str) -> str:
    return FEATURESET_LABELS.get(model, model)


def method_display(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


# ---------------------------------------------------------------------------
# Small shared helpers.
# ---------------------------------------------------------------------------
def _finish(ax, style):
    common = STYLE["common"]
    ax.spines[list(common["spines_off"])].set_visible(False)
    ax.grid(axis="y", alpha=common["grid_alpha"], linewidth=common["grid_linewidth"])
    ax.set_axisbelow(True)


def _set_ylim(ax, values, style):
    """Top from the data, bottom pinned at `ylim_bottom` (0 by default).

    A negative R^2 is a real result and the bar should hang below the axis line
    rather than the whole plot sliding down to accommodate it, so the floor does
    not move. `bar_label` still writes the number, anchored at the base.
    """
    finite = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    if not finite.size:
        return
    top = max(0.0, float(finite.max())) * style["ylim_headroom"]
    bottom = style["ylim_bottom"]
    if bottom is None:
        bottom = min(0.0, float(finite.min())) * style["ylim_headroom"]
    ax.set_ylim(bottom, top if top > bottom else bottom + 1.0)


def _label_bars(ax, containers):
    common = STYLE["common"]
    for container in containers:
        ax.bar_label(container, fmt=common["bar_label_format"],
                     fontsize=common["bar_label_fontsize"],
                     padding=common["bar_label_padding"])


def _flag_clipped_bars(ax, rotation=0):
    """Write the value of any bar that falls below the pinned axis floor.

    A negative R^2 is a real and interesting result -- the model did worse than
    predicting the mean -- and with the floor at zero its bar has nowhere to
    draw. Without this it reads as a missing result rather than a bad one.
    """
    common = STYLE["common"]
    bottom = ax.get_ylim()[0]
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if not np.isfinite(height) or height >= bottom:
                continue
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bottom),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", rotation=rotation,
                        fontsize=common["clipped_label_fontsize"],
                        color=common["clipped_label_color"])


def _starting_offsets(points, radius, names, overrides):
    """A first guess at each label's direction: away from the local crowd.

    Only a starting point. `_resolve_labels` measures the drawn text afterwards
    and pushes anything that still overlaps, which is the step that actually
    guarantees legibility -- character counts and guessed extents do not.
    """
    xy = np.asarray(points, dtype=float)
    if not len(xy):
        return []
    span = np.ptp(xy, axis=0)
    span[span == 0] = 1.0
    unit = (xy - xy.min(axis=0)) / span

    offsets = []
    for i, name in enumerate(names):
        if name in overrides:
            offsets.append(tuple(overrides[name]))
            continue
        delta = unit[i] - unit
        distance = np.linalg.norm(delta, axis=1)
        distance[i] = np.inf
        near = distance < 0.25
        direction = delta[near].sum(axis=0) if near.any() else np.array([0.0, 1.0])
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
        offsets.append((float(direction[0] * radius), float(direction[1] * radius)))
    return offsets


def _resolve_labels(ax, annotations, anchor_xy, fixed, iterations=120, pad=3.0):
    """Nudge annotations until their drawn boxes stop overlapping.

    Works on the real text extents rather than an estimate: the figure is drawn
    once, each label's bounding box is measured in display pixels, and the boxes
    are then pushed apart in that same space. Because these are *offset* labels,
    changing the offset translates the box exactly, so the boxes can be moved
    arithmetically without redrawing on every iteration.

    Labels also get pushed off the data points they would otherwise cover, and
    are kept inside the axes. Anything named in `STYLE[...]["offsets"]` is held
    where it was put -- a hand-placed label is a decision, not a starting guess.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    if not annotations:
        return

    boxes = np.array([[b.x0, b.y0, b.x1, b.y1] for b in
                      (a.get_window_extent(renderer) for a in annotations)], dtype=float)
    boxes[:, :2] -= pad
    boxes[:, 2:] += pad
    points = ax.transData.transform(np.asarray(anchor_xy, dtype=float))
    limits = ax.get_window_extent(renderer)
    movable = ~np.asarray(fixed, dtype=bool)
    shift = np.zeros((len(annotations), 2))

    def overlap(a, b):
        """Signed overlap on each axis; positive on both means they intersect."""
        return (min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1]))

    for _ in range(iterations):
        push = np.zeros((len(annotations), 2))

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                dx, dy = overlap(boxes[i], boxes[j])
                if dx <= 0 or dy <= 0:
                    continue
                # Separate along the cheaper axis, which keeps labels near their
                # own dot instead of sending them across the panel.
                centre_i = (boxes[i][:2] + boxes[i][2:]) / 2
                centre_j = (boxes[j][:2] + boxes[j][2:]) / 2
                if dx < dy:
                    step = np.array([dx / 2 + 1, 0.0])
                    sign = 1.0 if centre_i[0] >= centre_j[0] else -1.0
                else:
                    step = np.array([0.0, dy / 2 + 1])
                    sign = 1.0 if centre_i[1] >= centre_j[1] else -1.0
                push[i] += sign * step
                push[j] -= sign * step

        # A label sitting on top of a plotted dot.
        for i in range(len(annotations)):
            inside = ((points[:, 0] > boxes[i][0]) & (points[:, 0] < boxes[i][2]) &
                      (points[:, 1] > boxes[i][1]) & (points[:, 1] < boxes[i][3]))
            for point in points[inside]:
                centre = (boxes[i][:2] + boxes[i][2:]) / 2
                away = centre - point
                norm = np.linalg.norm(away)
                push[i] += (away / norm if norm > 1e-6 else np.array([0.0, 1.0])) * 4

        push[~movable] = 0.0
        if not np.abs(push).max() > 0.5:
            break
        boxes[:, :2] += push
        boxes[:, 2:] += push
        shift += push

        # Keep every box inside the axes, or the nudging walks labels off-panel.
        for i in np.flatnonzero(movable):
            correction = np.array([
                max(0.0, limits.x0 - boxes[i][0]) + min(0.0, limits.x1 - boxes[i][2]),
                max(0.0, limits.y0 - boxes[i][1]) + min(0.0, limits.y1 - boxes[i][3])])
            if correction.any():
                boxes[i][:2] += correction
                boxes[i][2:] += correction
                shift[i] += correction

    # Display pixels back to the typographic points the offsets are stated in.
    scale = 72.0 / figure.dpi
    for annotation, moved, can_move in zip(annotations, shift, movable):
        if not can_move or not moved.any():
            continue
        dx, dy = annotation.xyann
        annotation.set_position((dx + moved[0] * scale, dy + moved[1] * scale))


# ---------------------------------------------------------------------------
# Figure 1: pooled metric by feature section.
# ---------------------------------------------------------------------------
def pooled_by_featureset(summary: pd.DataFrame, dataset_name: str, metric: str = "r2",
                         methods=None, model_order=None):
    """Grouped bars: one group per feature section, one bar per method.

    `summary` is the exported summary table; only its `pooled_predictions` rows
    are used, so the heights are the pooled out-of-fold figures and do NOT equal
    the mean of that section's per-fold bars in the LOLO chart.

    Bars, not a line: the x axis is a handful of unrelated feature sections, and
    a line joining them would imply an ordering and a continuum that do not exist.
    """
    import matplotlib.pyplot as plt
    from .results import _model_order

    style = STYLE["pooled"]
    pooled = summary[summary.aggregation == "pooled_predictions"]
    if pooled.empty:
        raise ValueError("summary has no pooled_predictions rows to plot")

    order = list(model_order) if model_order else _model_order(pooled)
    table = pooled.pivot(index="model", columns="method", values=metric)
    table = table.reindex(index=[m for m in order if m in table.index])
    wanted = list(methods) if methods else [m for m in style["method_order"]
                                            if m in table.columns]
    wanted = [m for m in wanted if m in table.columns]
    if not wanted:
        wanted = list(table.columns)
    table = table[wanted]

    colors = [style["method_colors"].get(m, style["fallback_colors"][i % len(style["fallback_colors"])])
              for i, m in enumerate(table.columns)]
    labels = [method_display(m) for m in table.columns]
    display = table.copy()
    display.index = [featureset_label(m) for m in display.index]
    display.columns = labels

    ax = display.plot.bar(figsize=style["figsize"], width=style["bar_width"],
                          edgecolor=style["bar_edgecolor"],
                          linewidth=style["bar_edgewidth"], color=colors)
    ax.set(xlabel=style["xlabel"],
           ylabel=style["ylabel"].format(metric=metric_label(metric),
                                         unit=metric_unit(metric)))
    ax.set_title(style["title"].format(dataset=dataset_name, metric=metric_label(metric)),
                 fontsize=style["title_fontsize"])
    ax.tick_params(axis="x", rotation=style["tick_rotation"])
    ax.legend(title=style["legend_title"], fontsize=style["legend_fontsize"],
              title_fontsize=style["legend_fontsize"], frameon=False,
              loc=style["legend_loc"], bbox_to_anchor=style["legend_bbox"])
    _set_ylim(ax, display.to_numpy(dtype=float), style)
    _label_bars(ax, ax.containers)
    _flag_clipped_bars(ax, rotation=style["clipped_label_rotation"])
    _finish(ax, style)

    fig = ax.figure
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: the held-out-group summary.
# ---------------------------------------------------------------------------
def lolo_by_group(metrics_by_split: pd.DataFrame, dataset_name: str, metric: str = "r2",
                  group: str = "ligand", model_order=None):
    """One bar per held-out group, grouped by feature section.

    Each bar is that fold's score on that one held-out ligand's own rows, so it
    is measured against that ligand's own (smaller) variance -- these are NOT
    comparable with the pooled figures in `pooled_by_featureset`.
    """
    import matplotlib.pyplot as plt
    from .results import _model_order

    style = STYLE["lolo"]
    part = metrics_by_split[metrics_by_split.method == "lolo"]
    if part.empty:
        raise ValueError("No LOLO results available")

    table = part.pivot(index="reference_group", columns="model", values=metric)
    order = list(model_order) if model_order else _model_order(metrics_by_split)
    table = table.reindex(columns=[m for m in order if m in table.columns])
    table.columns = [featureset_label(m) for m in table.columns]

    palette = style["featureset_colors"]
    colors = [palette[i % len(palette)] for i in range(table.shape[1])]
    ax = table.plot.bar(figsize=style["figsize"], width=style["bar_width"], color=colors)
    ax.set(xlabel=style["xlabel"].format(group=group),
           ylabel=style["ylabel"].format(metric=metric_label(metric),
                                         unit=metric_unit(metric)))
    ax.set_title(style["title"].format(dataset=dataset_name, metric=metric_label(metric),
                                       group=group),
                 fontsize=style["title_fontsize"])
    ax.tick_params(axis="x", rotation=style["tick_rotation"])
    ax.legend(title=style["legend_title"], fontsize=style["legend_fontsize"],
              title_fontsize=style["legend_fontsize"], frameon=False,
              loc=style["legend_loc"], bbox_to_anchor=style["legend_bbox"])
    _set_ylim(ax, table.to_numpy(dtype=float), style)
    if style["bar_labels"]:
        _label_bars(ax, ax.containers)
    _flag_clipped_bars(ax, rotation=style["clipped_label_rotation"])
    _finish(ax, style)

    fig = ax.figure
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: where this dataset's ligands sit in the Kraken space.
# ---------------------------------------------------------------------------
def _scatter_panel(ax, population, used, names, xcol, ycol, xlabel, ylabel, title,
                   xlim, ylim, dataset_name, style, show_legend):
    ax.scatter(population[xcol], population[ycol], s=style["population_size"],
               c=style["population_color"], alpha=style["population_alpha"],
               edgecolors="none", zorder=1,
               label=style["population_label"].format(n=len(population)))
    ax.scatter(used[xcol], used[ycol], s=style["highlight_size"],
               c=style["highlight_color"], edgecolors=style["highlight_edgecolor"],
               linewidths=style["highlight_edgewidth"], zorder=3,
               label=style["highlight_label"].format(dataset=dataset_name, n=len(used)))

    if style["labels"] and len(used):
        points = used[[xcol, ycol]].to_numpy(dtype=float)
        offsets = _starting_offsets(points, style["label_radius"], list(names),
                                    style["offsets"])
        annotations = []
        for (x, y), name, (dx, dy) in zip(points, names, offsets):
            # ha/va are centred so the overlap solver can translate a box in any
            # direction without the text jumping relative to its own anchor.
            annotations.append(ax.annotate(
                name, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=style["label_fontsize"], fontweight=style["label_weight"],
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=style["label_box_alpha"]),
                arrowprops=dict(arrowstyle="-", lw=style["leader_linewidth"],
                                color=style["leader_color"], shrinkA=0,
                                shrinkB=style["leader_shrink"])))
        pending = (ax, annotations, points, [n in style["offsets"] for n in names])
    else:
        pending = None

    if style["zero_lines"]:
        ax.axhline(0, color="0.5", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.5", linewidth=0.8, zorder=0)

    ax.set_xlabel(xlabel, fontsize=style["axis_label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=style["axis_label_fontsize"])
    ax.set_title(title, fontsize=style["panel_title_fontsize"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(style["spine_linewidth"])
        ax.spines[side].set_color(style["spine_color"])
    ax.tick_params(width=style["tick_width"], length=style["tick_length"],
                   labelsize=style["tick_labelsize"], color=style["spine_color"])
    if show_legend:
        ax.legend(frameon=False, fontsize=style["legend_fontsize"],
                  loc=style["legend_loc"])
    return pending


def kraken_space(population: pd.DataFrame, used: pd.DataFrame, dataset_name: str,
                 name_column: str = "name"):
    """Two panels: PC1 vs PC2, and the two buried-volume descriptors.

    `population` is the Kraken reference -- every ligand in it, so the cloud is
    the same in every dataset's report and the panels can be read against each
    other. `used` is this dataset's own ligands, a subset of the same table with
    a name column for the point labels.
    """
    import matplotlib.pyplot as plt

    style = STYLE["kraken"]
    fig, axes = plt.subplots(1, 2, figsize=style["figsize"], constrained_layout=True)
    names = list(used[name_column]) if name_column in used.columns else [""] * len(used)

    pending = [
        _scatter_panel(axes[0], population, used, names,
                       style["pc_x"], style["pc_y"], style["pc_xlabel"],
                       style["pc_ylabel"], style["panel_titles"][0],
                       style["pc_xlim"], style["pc_ylim"],
                       dataset_name, style, show_legend=True),
        _scatter_panel(axes[1], population, used, names,
                       style["vbur_x"], style["vbur_y"], style["vbur_xlabel"],
                       style["vbur_ylabel"], style["panel_titles"][1],
                       style["vbur_xlim"], style["vbur_ylim"],
                       dataset_name, style, show_legend=False),
    ]

    fig.suptitle(style["suptitle"].format(dataset=dataset_name),
                 fontsize=style["suptitle_fontsize"], fontweight="bold")

    # Label overlaps are resolved against measured text boxes, so the axes must
    # have their final size first: draw once to let constrained layout settle,
    # then freeze it. Leaving the engine on would re-lay-out at savefig time and
    # move the axes out from under the offsets just computed.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for entry in pending:
        if entry is not None:
            _resolve_labels(*entry)
    return fig


# ---------------------------------------------------------------------------
# Saving.
# ---------------------------------------------------------------------------
def save(fig, directory, name: str, formats=("png",)) -> list:
    """Write one figure under `directory` in each format. Returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        fig.savefig(path, dpi=STYLE["common"]["dpi"], bbox_inches="tight",
                    facecolor=STYLE["common"]["facecolor"])
        written.append(path)
    return written
