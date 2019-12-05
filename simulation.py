import h5py as h5

import itertools
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.table import Table
from astropy import (coordinates as coord, units as u)
from astropy.coordinates.matrix_utilities import (matrix_product, rotation_matrix)
from tqdm import tqdm
from scipy import (optimize as op)
from astropy import constants

from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import LineCollection


import sys
sys.path.insert(0, "../code")
from mpl_utils import mpl_style
from utils import (approximate_ruwe, astrometric_excess_noise)

import twobody

plt.style.use(mpl_style)


np.random.seed(0)



def worker(j, **kwargs):
    ruwe, _ = approximate_ruwe(**kwargs)
    #aen, _ = astrometric_excess_noise(**kw)
    return (j, ruwe, np.nan)




# Gaia DR2 3425096028968232832
ra, dec = (92.95448746, 22.82574178)
gaia_ruwe = 1.45116



def simulate_orbit(t, P, m1, m2, f1=None, f2=None, e=0, t0=None,
                             omega=0*u.deg, i=0*u.deg, Omega=0*u.deg, 
                             origin=None, **kwargs):
    """
    Calculate the astrometric excess noise for a binary system with given
    properties that was observed at certain times from the given origin position.
    
    # TODO: There are a number of assumptions that we look over here

    :param t:
        The times that the system was observed.

    :param P:
        The period of the binary system.

    :param m1:
        The mass of the primary body.

    :param m2:
        The mass of the secondary body.

    :param f1: [optional]
        The flux of the primary body. If `None` is given then $M_1^{3.5}$ will
        be assumed.

    :param f2: [optional]
        The flux of the secondary body. If `None` is given then $M_2^{3.5}$ will
        be assumed.

    :param e: [optional]
        The eccentricity of the system (default: 0).

    # TODO: more docs pls
    """


    # TODO: Re-factor this behemoth by applying the weights after calculating positions?
    if f1 is None:
        f1 = m1.to(u.solMass).value**3.5
    if f2 is None:
        f2 = m2.to(u.solMass).value**3.5

    if t0 is None:
        t0 = Time('J2015.5')

    N = t.size
    
    # Compute orbital positions.
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        M1 = (2*np.pi * (t.tcb - t0.tcb) / P).to(u.radian)
        # Set secondary to have opposite phase.
        M2 = (2*np.pi * (t.tcb - t0.tcb) / P - np.pi).to(u.radian)
        
    # eccentric anomaly
    E1 = twobody.eccentric_anomaly_from_mean_anomaly(M1, e)
    E2 = twobody.eccentric_anomaly_from_mean_anomaly(M2, e)

    # mean anomaly
    F1 = twobody.true_anomaly_from_eccentric_anomaly(E1, e)
    F2 = twobody.true_anomaly_from_eccentric_anomaly(E2, e)

    # Calc a1/a2.
    m_total = m1 + m2
    a = twobody.P_m_to_a(P, m_total)
    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    r1 = (a1 * (1. - e * np.cos(E1))).to(u.au).value
    r2 = (a2 * (1. - e * np.cos(E2))).to(u.au).value

    # Calculate xy positions in orbital plane.
    x = np.vstack([
        r1 * np.cos(F1),
        r2 * np.cos(F2),
    ]).value
    y = np.vstack([
        r1 * np.sin(F1),
        r2 * np.sin(F2)
    ]).value

    # Calculate photocenter in orbital plane.
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    x, y = np.vstack([w @ x, w @ y])
    z = np.zeros_like(x)

    # Calculate photocenter velocities in orbital plane.
    fac = (2*np.pi * a / P / np.sqrt(1 - e**2)).to(u.au/u.s).value
    vx = np.vstack([
        -fac * np.sin(F1),
        -fac * np.sin(F2)
    ]).value
    vy = np.vstack([
        fac * (np.cos(F1) + e),
        fac * (np.cos(F2) + e)
    ]).value
    vx, vy = np.vstack([w @ vx, w @ vy])
    vz = np.zeros_like(vx)

    # TODO: handle units better w/ dot product
    x, y, z = (x * u.au, y * u.au, z * u.au)
    vx, vy, vz = (vx * u.au/u.s, vy * u.au/u.s, vz * u.au/u.s)
    
    xyz = coord.CartesianRepresentation(x=x, y=y, z=z)
    vxyz = coord.CartesianDifferential(d_x=vx, d_y=vy, d_z=vz)
    xyz = xyz.with_differentials(vxyz)

    vxyz = xyz.differentials["s"]
    xyz = xyz.without_differentials()

    # Construct rotation matrix from orbital plane system to reference plane system.
    R1 = rotation_matrix(-omega, axis='z')
    R2 = rotation_matrix(i, axis='x')
    R3 = rotation_matrix(Omega, axis='z')
    Rot = matrix_product(R3, R2, R1)

    # Rotate photocenters to the reference plane system.
    XYZ = coord.CartesianRepresentation(matrix_product(Rot, xyz.xyz))
    VXYZ = coord.CartesianDifferential(matrix_product(Rot, vxyz.d_xyz))
    XYZ = XYZ.with_differentials(VXYZ)

    barycenter = twobody.Barycenter(origin=origin, t0=t0)
    kw = dict(origin=barycenter.origin)
    rp = twobody.ReferencePlaneFrame(XYZ, **kw)

    # Calculate the ICRS positions.
    icrs_cart = rp.transform_to(coord.ICRS).cartesian
    icrs_pos = icrs_cart.without_differentials()
    icrs_vel = icrs_cart.differentials["s"]

    bary_cart = barycenter.origin.cartesian
    bary_vel = bary_cart.differentials["s"]

    dt = t - barycenter.t0
    dx = (bary_vel * dt).to_cartesian()

    pos = icrs_pos + dx
    vel = icrs_vel + bary_vel

    icrs = coord.ICRS(pos.with_differentials(vel))

    x = (icrs.ra - np.mean(icrs.ra)).to(u.mas)
    y = (icrs.dec - np.mean(icrs.dec)).to(u.mas)



    fig, ax = plt.subplots()
    scat = ax.scatter(x.value, y.value, c=t.mjd)

    cbar = plt.colorbar(scat)


    raise a



# From the paper (their preferred values on distance, etc).
distance = 4.230 * u.kpc
eccentricity = 0.03 # pm 0.01
period = 78.9 * u.day # \pm 0.3 * u.day
m1 = 68 * u.solMass # (+11, -13) from their paper (abstract)
m2 = 8.2 * u.solMass # (+0.9, -1.2) from their paper
f1, f2 = (0, 1)
i = 15 * u.deg # bottom right of page 2 of their paper

omega = 0 * u.deg # doesn't matter. thanks, central limit thereom!
radial_velocity = 0 * u.km/u.s # doesn't matter here

# From Gaia, unless otherwise defined:
origin_kwds = dict(ra=92.95448 * u.deg,
                   dec=22.82574 * u.deg,
                   distance=distance,
                   pm_ra_cosdec=-0.0672 * u.mas/u.yr,
                   pm_dec=-1.88867* u.mas/u.yr,
                   radial_velocity=radial_velocity)

origin = coord.ICRS(**origin_kwds)
kwds = dict(i=i, omega=omega, origin=origin)

kwds.update(P=period,
            m1=m1,
            f1=f1,
            m2=m2,
            f2=f2,
            distance=origin.distance) # ignored by astrometric_excess_noise but needed for approximating functions





# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')
observing_span = obs_end - obs_start


gost = Table.read("gost_21.3.1_652094_2019-12-05-02-26-18.csv")
t = Time(gost["ObservationTimeAtGaia[UTC]"])




# Assume observed at uniformly random times.
#t = obs_start + np.linspace(0, 1, astrometric_n_obs) * (obs_end - obs_start)

kwds.update(t=t)
#kwds.update(pm_ra_cosdec=0*u.mas/u.yr, pm_dec=0*u.mas/u.yr)
#simulate_orbit(**kwds)





lb1_approximate_ruwe, meta_ruwe = approximate_ruwe(**kwds)
lb1_aen, meta_aen = astrometric_excess_noise(**kwds)

print(f"Reported RUWE by Gaia DR2 = {gaia_ruwe:.2f}")
print(f"Approximated RUWE = {lb1_approximate_ruwe:.2f}")
print(f"Estimated AEN = {lb1_aen:.2f}")


"""
Do Approximate Bayesian Computation.

Free parameters:
- inclination angle (0, 90)
- m1 (1, 70)
- m2 (0.7, 70)
- distance (1, 5)
"""

processes = 50
#n_repeats = 1

#0.77,
#5.37

cost_factor = 1

i_bins = np.linspace(0, 90, cost_factor * 30) # u.deg
m1_bins = np.linspace(1, 70, cost_factor * 70) # u.solMass
m2_bins = np.linspace(1, 15, cost_factor * 30) # u.solMass
distance_bins = np.linspace(2, 5, cost_factor * 30) # u.kpc

pool = mp.Pool(processes=processes)

results = []

grid = np.array(list(itertools.product(i_bins, m1_bins, m2_bins, distance_bins)))

grid_ruwe = np.zeros(grid.shape[0])
grid_aen = np.zeros(grid.shape[0])


for j, (i, m1, m2, distance) in enumerate(tqdm(grid, desc="Hunting")):

    okw = origin_kwds.copy()
    okw.update(distance=distance * u.kpc)

    origin = coord.ICRS(**okw)
    kwds.update(i=i * u.deg,
                m1=m1 * u.solMass,
                m2=m2 * u.solMass,
                origin=origin,
                distance=origin.distance)

    if processes == 1:
        grid_ruwe[j], _ = approximate_ruwe(**kwds)
        grid_aen[j], _ = astrometric_excess_noise(**kwds)

    else:
        results.append(pool.apply_async(worker, (j, ), kwds))


if processes > 1:

    for each in tqdm(results, desc="Collecting"):
        j, ruwe, aen = each.get(timeout=1)
        grid_ruwe[j] = ruwe
        grid_aen[j] = aen


    pool.close()
    pool.join()


#grid = grid[:, :-1]

# Take mean from repeats?
# TODO:
target_ruwe = gaia_ruwe
ruwe_tolerance = 0.1
mask = np.abs(target_ruwe - grid_ruwe) < ruwe_tolerance

X = grid[mask]

from corner import corner

fig = corner(X, labels=(r"$i$", r"$M_1$", r"$M_2$", r"$d$"),
             range=list(zip(np.min(grid, axis=0), np.max(grid, axis=0))))
#fig.savefig("abc.png", dpi=300)
    

raise a



