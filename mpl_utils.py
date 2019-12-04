
# A matplotlib style based on the gala package by @adrn:
# github.com/adrn/gala

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

mpl_style = {

    # Lines
    'lines.linewidth': 1.7,
    'lines.antialiased': True,
    'lines.marker': '.',
    'lines.markersize': 5.,

    # Patches
    'patch.linewidth': 1.0,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#CCCCCC',
    'patch.antialiased': True,

    # images
    'image.origin': 'upper',

    # colormap
    'image.cmap': 'viridis',

    # Font
    'font.size': 12.0,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.latex.preview': True,
    'axes.unicode_minus': False,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,

    # Ticks
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.color': '#333333',
    'xtick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.color': '#333333',
    'ytick.direction': 'in',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Legend
    'legend.fancybox': True,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [6, 6],
    'figure.facecolor': '1.0',
    'figure.edgecolor': '0.50',
    'figure.subplot.hspace': 0.5,

    # Other
    'savefig.dpi': 300,
}




def corner_scatter(X, label_names=None, show_ticks=False, fig=None, figsize=None,
                   **kwargs):
    """
    Make a corner plot where the data are shown as scatter points in each axes.

    :param X:
        The data, :math:`X`, which is expected to be an array of shape
        [n_samples, n_features].

    :param label_names: [optional]
        The label names to use for each feature.

    :param show_ticks: [optional]
        Show ticks on the axes.

    :param fig: [optional]
        Supply a figure (with [n_features, n_features] axes) to plot the data.

    :param figsize: [optional]
        Specify a size for the figure. This parameter is ignored if a `fig` is
        supplied.

    :returns:
        A figure with a corner plot showing the data.
    """

    N, D = X.shape
    assert N > D, "Stahp doing it wrong"
    K = D

    if fig is None:
        if figsize is None:
            figsize = (2 * K, 2 * K)
        fig, axes = plt.subplots(K, K, figsize=figsize)
    
    axes = np.array(fig.axes).reshape((K, K)).T

    kwds = dict(s=1, c="tab:blue", alpha=0.5, rasterized=True)
    kwds.update(kwargs)
    
    for i, x in enumerate(X.T):
        for j, y in enumerate(X.T):
            if j == 0: continue

            try:
                ax = axes[i, j]

            except:
                continue


            if i > j:
                ax.set_visible(False)
                continue

            elif i == j:
                ax.set_facecolor("#eeeeee")
                continue

            ax.scatter(x, y, **kwds)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            else:
                if not ax.is_last_row():
                    ax.set_xticklabels([])
                if not ax.is_first_col():
                    ax.set_yticklabels([])


            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])

    fig.tight_layout()
    
    return fig


def plot_histogram_steps(ax, x_bins, y, y_err, lw=1, **kwargs):

    xx = np.array(x_bins).repeat(2)[1:]
    xstep = np.repeat((x_bins[1:] - x_bins[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1]) - xstep/2.0

    yy = np.array(y).repeat(2)

    #xbc = x_bins[:-1] + 0.5 * np.diff(x_bins)

    _ = ax.plot(xx, yy, '-', lw=lw, **kwargs)
    ax.errorbar(x_bins, y, y_err, fmt="none", lw=lw, capsize=0, ecolor=_[0].get_color())

    return (xx, yy)

def plot_binned_statistic(x, y, z, bins=100, function=np.nanmedian,
                          xlabel=None, ylabel=None, zlabel=None,
                          ax=None, colorbar=False, figsize=(8, 8),
                          vmin=None, vmax=None, min_entries_per_bin=None,
                          subsample=None, mask=None, full_output=False, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    finite = np.isfinite(x * y * z)
    if mask is not None:
        finite *= mask
    if subsample is not None:
        idx = np.where(finite)[0]
        if subsample < 1:
            subsample *= idx.size
        if int(subsample) > idx.size:
            finite = idx
        else:
            finite = np.random.choice(idx, int(subsample), replace=False)
    
    H, xedges, yedges, binnumber = binned_statistic_2d(
        x[finite], y[finite], z[finite],
        statistic=function, bins=bins)

    if min_entries_per_bin is not None:
        if function != "count":
            H_count, xedges, yedges, binnumber = binned_statistic_2d(
                x[finite], y[finite], z[finite],
                statistic="count", bins=bins)

        else:
            H_count = H

        H[H_count < min_entries_per_bin] = np.nan


    if (vmin is None or vmax is None) and "norm" not in kwargs:
        vmin_default, med, vmax_default = np.nanpercentile(H, kwargs.pop("norm_percentiles", [5, 50, 95]))
        if vmin is None:
            vmin = vmin_default
        if vmax is None:
            vmax = vmax_default
    
    imshow_kwds = dict(
        vmin=vmin, vmax=vmax,
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno",
        interpolation="bilinear")
    imshow_kwds.update(kwargs)
    
    image = ax.imshow(H.T, **imshow_kwds)
    if colorbar:
        cbar = plt.colorbar(image, ax=ax)
        if zlabel is not None:
            cbar.set_label(zlabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    #fig.tight_layout()

    return (fig, image) if full_output else fig