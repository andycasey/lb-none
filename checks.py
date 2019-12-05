import h5py as h5

import itertools
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import (coordinates as coord, units as u)
from astropy.coordinates.matrix_utilities import (matrix_product, rotation_matrix)
from tqdm import tqdm
from scipy import (optimize as op)
from astropy import constants

from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import LineCollection


import sys
#sys.path.insert(0, "../code")
from mpl_utils import mpl_style
from utils import (approximate_ruwe, astrometric_excess_noise, salpeter_imf)

import twobody

plt.style.use(mpl_style)


np.random.seed(0)



# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')
observing_span = obs_end - obs_start


# Put the sky position something where it will not wrap...
# Also you should set RV = 0 and proper motion as zero because here we assume that
# the contribution of those factors has been accounted for by Gaia
origin = coord.ICRS(ra=0.1 * u.deg,
                    dec=0.1 * u.deg,
                    distance=1000 * u.pc,
                    pm_ra_cosdec=0 * u.mas/u.yr,
                    pm_dec=0 * u.mas/u.yr,
                    radial_velocity=0 * u.km/u.s)
kwds = dict(i=0 * u.deg, omega=0 * u.deg, origin=origin)

"""
if kwds["i"] > (0 * u.deg):
    print("Warning: approximate AEN calculations currently ignore inclination.")

if kwds["pm_ra_cosdec"] > (0 * u.mas/u.yr) \
or kwds["pm_dec"] > (0 * u.mas/u.yr):
    print("Warning: proper motion should be zero as we assume Gaia has accounted for this")
"""  





N_repeats = 100
Ms = salpeter_imf(N_repeats, 2.35, 0.1, 100) * u.solMass

# Get the number of astrometric obs.
with h5.File("../trex/data/sources.hdf5", "r") as sources:
    astrometric_n_good_obs_al = np.random.choice(sources["sources"]["astrometric_n_good_obs_al"], N_repeats)

#astrometric_n_good_obs_al = (225 * np.ones(N_repeats)).astype(int)


BURN_FORESTS = True
PROCESSES = 10

q_bins, P_bins = (10, 10)
Ps = np.logspace(-1.5, 2.5, P_bins)
qs = np.linspace(0.1, 1, q_bins)

qPs = np.array(list(itertools.product(qs, Ps)))

ruwe = np.zeros((qPs.shape[0], N_repeats), dtype=float)
approx_ruwe = np.zeros((qPs.shape[0], N_repeats), dtype=float)

extras = np.zeros((qPs.shape[0], N_repeats, 4), dtype=float)

assert max(Ps) < observing_span.to(u.day).value

def _mp_approx_ruwe(i, j, kw):
    v, meta = approximate_ruwe(**kw)
    return (i, j, v)

def _mp_actual_aen(i, j, kw):
    v, meta = astrometric_excess_noise(**kw)
    return (i, j, v)#v.to(u.mas).value)


if PROCESSES > 1:
    pool = mp.Pool(processes=PROCESSES)

    p_approx = []
    p_actual = []

print("Simulating binary systems...")

for i, (q, P) in enumerate(tqdm(qPs)):

    P = P * u.day

    for j, (m1, N) in enumerate(zip(Ms, astrometric_n_good_obs_al)):
        
        t = obs_start + np.linspace(0, 1, N) * (obs_end - obs_start)
        m2 = q * m1

        kw = kwds.copy()
        kw.update(t=t, 
                  P=P,
                  m1=m1, m2=m2,
                  f1=m1.to(u.solMass).value**3.5, f2=m2.to(u.solMass).value**3.5,
                  # This gets ignored by astrometric_excess_noise but is needed for approximations
                  distance=origin.distance)

        v = [P.value, q, m1.value, m2.value]
        extras[i, j, :len(v)] = v

        if PROCESSES > 1:
            p_approx.append(pool.apply_async(_mp_approx_ruwe, (i, j, kw)))

            if BURN_FORESTS:
                p_actual.append(pool.apply_async(_mp_actual_aen, (i, j, kw)))

        else:
            #approx_rms, approx_meta = approximate_ruwe(**kw)
            r, approx_meta = approximate_ruwe(**kw)
            # TODO: Don't take RMS == AEN! They are different!
            #approx_aen[i, j] = approx_rms.to(u.mas).value        

            approx_ruwe[i, j] = r
           
            if BURN_FORESTS:
                actual_aen, actual_meta = astrometric_excess_noise(**kw)
                #ruwe[i, j] = np.sqrt(actual_aen.to(u.mas).value/(len(kw["t"]) - 2))
                ruwe[i, j] = actual_aen#.to(u.deg).value
    
        foo, bar = ruwe[i, j], approx_ruwe[i, j]
        
                
if PROCESSES > 1:

    print("Collecting results")
    for each in tqdm(p_approx):
        i, j, v = each.get(timeout=1)
        approx_ruwe[i, j] = v

    for each in p_actual:
        i, j, v = each.get(timeout=1)
        ruwe[i, j] = v

    pool.close()
    pool.join()


if not BURN_FORESTS:
    print("Using AEN approximations")
    #aen = approx_aen
    ruwe = approx_ruwe

else:
    print("Plotting using expensive AEN -- doing comparisons")

    fig, ax = plt.subplots()
    ax.scatter(ruwe.flatten(), approx_ruwe.flatten(), s=1)

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))
    ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1, ms=0, lw=1)
    ax.set_xlabel(r"{RUWE expensive}")
    ax.set_ylabel(r"{RUWE cheap}")

    options = [
        ("P", True),
        ("q", False),
        ("M1", False),
        ("M2", False)
    ]

    abs_diff = np.abs(ruwe - approx_ruwe)

    for i, (label, is_norm) in enumerate(options):


        kw = dict(s=1, c=extras[:, :, i].flatten())
        if is_norm:
            kw.update(norm=LogNorm())

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        ax = axes[0]
        ax.scatter(ruwe.flatten(), approx_ruwe.flatten(), **kw)
        ax.set_xlabel(r"{RUWE expensive}")
        ax.set_ylabel(r"{RUWE cheap}")

        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))
        ax.plot(limits, limits, c="#666666", linestyle=":", zorder=-1, ms=0, lw=1)

        ax = axes[1]
        scat = ax.scatter(ruwe.flatten(), abs_diff.flatten(), **kw)
        ax.semilogy()
        ax.set_ylim(10**-17, 10**-0)

        cbar = plt.colorbar(scat)
        cbar.set_label(label)

        ax.set_xlabel(r"{RUWE expensive}")
        ax.set_ylabel(r"$|\Delta|$")


if not BURN_FORESTS:

    mean_ruwe = np.mean(ruwe, axis=1).reshape((q_bins, P_bins))

    # Plot per Q first.
    cmap = cm.viridis(qs)
    fig, ax = plt.subplots()

    lc = LineCollection([np.column_stack([Ps, mr]) for mr in mean_ruwe])

    lc.set_array(np.asarray(qs))
    ax.add_collection(lc)
    ax.autoscale()


    #    print(i, q, np.mean(mean_aen[i]))
    cbar = plt.colorbar(lc)
    cbar.set_label(r"$q$")



    ax.set_xlabel(r"{period / days}$^{-1}$")
    ax.set_ylabel(r"{RUWE}")
    ax.semilogx()

    v = (obs_end - obs_start).to(u.day).value
    axvline_kwds = dict(c="#666666", zorder=-1, lw=1, ms=1)
    ax.axvline(v, linestyle=":", **axvline_kwds)
    ax.axvline(2 * v, linestyle="-", **axvline_kwds)


    print(kwds)


    qm, Pm = np.meshgrid(qs, Ps)

    contourf_kwds = dict(cmap="magma", norm=LogNorm(), levels=None)

    fig, ax = plt.subplots()
    im = ax.contourf(Ps, qs, mean_ruwe, **contourf_kwds)
    ax.semilogx()


    ax.set_xlabel(r"{period / days}$^{-1}$")
    ax.set_ylabel(r"$q$")

    cbar = plt.colorbar(im)
    cbar.set_label(r"{RUWE / mas}")

    fig.tight_layout()

    axvline_kwds.update(zorder=10)

    ax.axvline(v, linestyle=":", **axvline_kwds)
    ax.axvline(2 * v, linestyle="-", **axvline_kwds)


    print(kwds)


    # Let's do a simple thing about detector efficiency.
    # Let's assume anything more than 0.1 umas RMS is a detected binary.
    #detected = (aen >= (0.1 * u.mas).value).astype(int)
    detected = (ruwe >= 1.5).astype(int)

    de = np.mean(detected, axis=1).reshape((q_bins, P_bins))


    N_levels = 10
    contourf_kwds = dict(cmap="Blues", norm=LogNorm(), levels=np.logspace(-1, 0, N_levels + 1))
    contourf_kwds = dict(cmap="Blues",levels=np.linspace(0, 1, N_levels + 1))


    fig, ax = plt.subplots()
    im = ax.contourf(Ps, qs, de, **contourf_kwds)
    ax.semilogx()

    cbar = plt.colorbar(im)
    cbar.set_label(r"{detection efficiency}")

    ax.set_xlabel(r"{period / days}$^{-1}$")
    ax.set_ylabel(r"$q$")

    fig.tight_layout()

    # Show the observing span and twice that.
    axvline_kwds = dict(c="#000000", zorder=10, lw=1, ms=0)
    ax.axvline(observing_span.to(u.day).value, linestyle=":", **axvline_kwds)
    ax.axvline(2 * observing_span.to(u.day).value, linestyle="-", **axvline_kwds)

