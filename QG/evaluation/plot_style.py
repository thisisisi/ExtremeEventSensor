#!/usr/bin/env python3


"""Shared plotting defaults and external-legend layout."""


import numpy as np


import matplotlib as mpl


mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,

    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,

    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,

    "legend.frameon": False,
    "lines.linewidth": 2.4,
    "lines.solid_capstyle": "round",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})





def _unique_legend_items(axs):
    items = {}

    for ax in np.ravel(axs):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in items:
                items[label] = handle

    return items

def _add_external_legend(fig, axs, position="top"):
    items = _unique_legend_items(axs)

    if not items:
        return

    if position == "right":
        fig.legend(
            items.values(),
            items.keys(),
            loc="center left",
            bbox_to_anchor=(0.805, 0.5),
            handlelength=3.2,
            handletextpad=0.8,
            labelspacing=0.75,
            borderaxespad=0.0,
            frameon=False,
        )
        fig.tight_layout(rect=(0.065, 0.07, 0.79, 0.98))

    elif position == "top":
        fig.legend(
            items.values(),
            items.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=min(3, len(items)),
            handlelength=3.0,
            handletextpad=0.7,
            columnspacing=1.25,
            frameon=False,
        )
        fig.tight_layout(rect=(0.065, 0.07, 0.98, 0.90))

    else:
        raise ValueError("legend_position must be 'right' or 'top'")


# ------------------ DRIVER ------------------ #


# (sig, n_ens, ntime, strategy, model_type, Nepochs, dt_ml, basis_type)
    # plot_model_comparison(configs_to_plot, plot_type="spectrum")
    # plot_model_comparison(configs_to_plot, plot_type="zonal")
