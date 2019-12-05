
"""
"""


import numpy as np
import matplotlib.pyplot as plt

#from mpl_utils import mpl_style
#plt.style.use(mpl_style)

import vaex
import h5py as h5


from matplotlib.ticker import MaxNLocator




gaia = h5.File("../trex/data/5482.hdf5", "r")


l = gaia["sources/l"][:]
b = gaia["sources/b"][:]

ra_parallax_corr = gaia["sources/ra_parallax_corr"][:]
dec_parallax_corr = gaia["sources/dec_parallax_corr"][:]

v = ra_parallax_corr

finite = np.isfinite(l * b * v)

df = vaex.from_arrays(source_id=gaia["sources/source_id"][:][finite],
                    l=l[finite], 
                      b=b[finite], 
                      ra_parallax_corr=ra_parallax_corr[finite],
                      dec_parallax_corr=dec_parallax_corr[finite])


colormap = "coolwarm"


ra_parallax_corr = 0.53965366
l, b = (188.23526379595597, 2.050886146198157)
within = 2

mask = (np.abs(gaia["sources/l"][:] - l) <= within) \
     * (np.abs(gaia["sources/b"][:] - b) <= within)


vmin = -0.7
vmax = +0.7


def plot_mean_gaia_measurement(df, parameter_name, colorbar_label=None, **kwargs):

    fig, ax = plt.subplots()

    kwds = dict(healpix_level=6, 
                figsize=(8, 8), colorbar=True,
                colormap=colormap,
                grid_limits=(vmin, vmax))
    kwds.update(kwargs)

    df.healpix_plot(what=f"mean({parameter_name})", **kwds)

    ax = plt.gca() # to be sure.
    # Remove ugly text
    ax.texts[0].set_visible(False)

    # Change colorbar text
    if colorbar_label is not None:
        cbar = ax.get_images()[0].colorbar
        cbar.ax.texts[0].set_visible(False)
        cbar.set_label(colorbar_label)

    fig = ax.figure
    fig.tight_layout()
    return fig



fig_ra = plot_mean_gaia_measurement(df, "ra_parallax_corr", colorbar_label="ra_parallax_corr", figsize=(8, 4))

#raise a
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#ax_inset = inset_axes(fig_ra.axes[-1], width=1, height=1, loc=0)



scale = fig_ra.get_figwidth()/fig_ra.get_figheight()
w = 0.25
h = scale * w
space = 0.05
ax_inset = fig_ra.add_axes([space*2, 1 - space - h, w, h])

scat = ax_inset.scatter(gaia["sources/l"][:][mask],
                      gaia["sources/b"][:][mask],
                      c=gaia["sources/ra_parallax_corr"][:][mask],
                      s=1, cmap=colormap, vmin=vmin, vmax=vmax)

ax_inset.scatter([l], [b], c=[ra_parallax_corr], s=30, edgecolor="k", lw=1, vmin=-1, vmax=1, cmap=colormap)

ax_inset.xaxis.set_major_locator(MaxNLocator(3))
ax_inset.yaxis.set_major_locator(MaxNLocator(3))

ax_inset.set_xlim(l - within, l + within)
ax_inset.set_ylim(b - within, b + within)


plt.show()
plt.draw()

ax_inset.set_xlabel(r"$l^\circ$")
ax_inset.set_ylabel(r"$b^\circ$")

fig_ra.savefig("gaia_ra_parallax_corr.pdf", dpi=300)





dec_parallax_corr = -0.61868155

fig_dec = plot_mean_gaia_measurement(df, "dec_parallax_corr", colorbar_label="dec_parallax_corr", figsize=(8, 4))

#raise a
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#ax_inset = inset_axes(fig_dec.axes[-1], width=1, height=1, loc=0)

scale = fig_dec.get_figwidth()/fig_dec.get_figheight()
w = 0.25
h = scale * w
space = 0.05
ax_inset = fig_dec.add_axes([space*2, 1 - space - h, w, h])

scat = ax_inset.scatter(gaia["sources/l"][:][mask],
                      gaia["sources/b"][:][mask],
                      c=gaia["sources/dec_parallax_corr"][:][mask],
                      s=1, cmap=colormap, vmin=vmin, vmax=vmax)

ax_inset.scatter([l], [b], c=[dec_parallax_corr], s=30, edgecolor="k", lw=1, vmin=-1, vmax=1, cmap=colormap)

ax_inset.xaxis.set_major_locator(MaxNLocator(3))
ax_inset.yaxis.set_major_locator(MaxNLocator(3))

ax_inset.set_xlim(l - within, l + within)
ax_inset.set_ylim(b - within, b + within)


plt.show()
plt.draw()

ax_inset.set_xlabel(r"$l^\circ$")
ax_inset.set_ylabel(r"$b^\circ$")

fig_dec.savefig("gaia_dec_parallax_corr.pdf", dpi=300)



