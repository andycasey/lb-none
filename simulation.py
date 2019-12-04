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
sys.path.insert(0, "../code")
from mpl_utils import mpl_style

import twobody

plt.style.use(mpl_style)


np.random.seed(0)



# Functions:
# (1) Approximate the astrometric RMS by ignoring inclination, sky distortian, etc.
# (2) Exactly calculate the astrometric RMS.

def approximate_ruwe(t, P, m1, m2, distance, f1=None, f2=None, t0=None, 
                     i=0*u.deg, **kwargs):
    """
    Approximate the on-sky astrometric excess noise for a binary system with the
    given system parameters at a certain distance.

    This approximating function ignores the following effects:

    (1) The distortions that arise due to sky projections.
    (2) Inclination effects.
    (3) Omega effects.

    In part it also assumes:

    (1) The times were observed pseudo-randomly.
    (2) The orbit is fully sampled.

    :param t:
        The times that the system was observed.

    :param P:
        The period of the binary system.

    :param m1:
        The mass of the primary star.

    :param m2:
        The mass of the secondary system.

    :param distance:
        The distance from the observer to the center of mass of the binary
        system.

    :param f1: [optional]
        The flux of the primary star. If `None` is given then this is assumed to
        be $m_1^{3.5}$.

    :param f2: [optional]
        The flux of the secondary. If `None` is given then this is assumed to be
        $m_2^{3.5}$.

    :returns:
        A two-part tuple containing the root-mean-squared deviations in on-sky
        position (in units of milliarcseconds), and a dictionary containing meta
        information about the binary system.
    """

    if f1 is None:
        f1 = m1.to(u.solMass).value**3.5
    if f2 is None:
        f2 = m2.to(u.solMass).value**3.5

    if t0 is None:
        t0 = Time('J2015.5')

    m_total = m1 + m2
    w = np.array([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    w1, w2 = (w[0], w[1])

    # TODO: replace this with integral!
    dt = (t - t0).to(u.day)
    phi = (2 * np.pi * dt / P).value
    N = phi.size

    dx = a1 * w1 * np.cos(phi) + a2 * w2 * np.cos(phi + np.pi)
    dy = a1 * w1 * np.sin(phi) + a2 * w2 * np.sin(phi + np.pi)

    planar_rms_in_au = np.sqrt(np.sum((dx - np.mean(dx))**2 + (dy - np.mean(dy))**2)/N).value

    # Need some corrections for when the period is longer than the observing timespan, and the
    # inclination angle is non-zero.

    # For this it really depends on what t0/Omega is: if you see half the orbit in one phase or
    # another...
    # TODO: this requires a thinko.
    

    """
    Approximate given some inclination angle.
    At zero inclination, assume circle on sky such that:
    
        rms = sqrt(ds^2 + ds^2) = sqrt(2ds^2)

    and 
        
        ds = np.sqrt(0.5 * rms^2)

    Now when inclined (even at 90) we still get ds + contribution:

        rms_new = sqrt(ds^2 + (cos(i) * ds)^2)
    """

    ds = np.sqrt(0.5 * planar_rms_in_au**2)
    rms_in_au = np.sqrt(ds**2 + (np.cos(i) * ds)**2)
    rms_in_mas = (rms_in_au * u.au / distance).to(u.mas, equivalencies=u.dimensionless_angles())

    # Intrinsic error on position in one direction is.
    # These are the final values. The individual epochs are probably about a 10th of this.
    intrinsic_ra_error = 0.029 # mas
    intrinsic_dec_error = 0.026 # mas

    intrinsic_ra_error /= 10
    intrinsic_dec_error /= 10

    chi2 = N * rms_in_mas.to(u.mas).value**2 / (intrinsic_ra_error**2 + intrinsic_dec_error**2)

    # sqrt(2) from approximating rms in one dimension instead of 2
    approx_ruwe = np.sqrt(2) * np.sqrt(chi2/(N - 2))
    
    meta = dict(weights=w,
                a=a,
                a1=a1,
                a2=a2,
                w1=w1,
                w2=w2,
                phi=phi,
                dx=dx,
                dy=dy,
                rms_in_au=rms_in_au)

    return (approx_ruwe, meta)


def astrometric_excess_noise(t, P, m1, m2, f1=None, f2=None, e=0, t0=None,
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

    positions = np.array([icrs.ra.deg, icrs.dec.deg])

    mean_position = np.mean(positions, axis=1)
    assert mean_position.size == 2
    '''
    intrinsic_ra_error = 0.029 # mas
    intrinsic_dec_error = 0.026 # mas

    intrinsic_ra_error /= 10
    intrinsic_dec_error /= 10

    chi2 = N * rms_in_mas.to(u.mas).value**2 / np.sqrt(intrinsic_ra_error**2 + intrinsic_dec_error**2)

    approx_ruwe = np.sqrt(chi2/(N - 2))
    '''
    
    
    intrinsic_ra_error = 0.029 # mas
    intrinsic_dec_error = 0.026 # mas

    intrinsic_ra_error /= 10
    intrinsic_dec_error /= 10

    
    # Calculate on sky RMS.

    astrometric_rms = np.sqrt(np.sum((positions.T - mean_position)**2)/N)
    astrometric_rms *= u.deg
    
    diff = ((positions.T - mean_position) * u.deg).to(u.mas)
    #chi2 = diff**2 / (intrinsic_ra_error**2 + intrinsic_dec_error**2)
    ruwe = np.sqrt(np.sum((diff.T[0]/intrinsic_ra_error)**2 + (diff.T[1]/intrinsic_dec_error)**2)/(N-2)).value
        
    meta = dict()
    return (ruwe, meta)
    #return (astrometric_rms, meta)



def worker(j, k, kw):
    ruwe, _ = approximate_ruwe(**kw)
    #aen, _ = astrometric_excess_noise(**kw)
    return (j, k, ruwe, np.nan)

if __name__ == "__main__":

    # https://www.nature.com/articles/s41586-019-1766-2.epdf?referrer_access_token=8Fl250M3P31dj7_i4DY-KNRgN0jAjWel9jnR3ZoTv0NNeBQ9KXA2F0hGDxVf7ameB3I4LFBZPLQcHSsiKhInFxdXaUZsIEFwvayNCt7jvJbjLHplD4YnhSIGvLbOd8FKDMECCvQmMBvgPPiOkfI1PcLnGxSmuL_O3iVdLBEwbVIP5nTNASsHZ1U57wQioXm8TvFUwvO_S6yuT7HlyrbQGci5l-o6u9gxETegsz3mFiryLh8Qavyt4GRZd2g1CHNPU8JPz2O_TO5J1OrLZ9wxkXTzuRvBM50cBegxHO_PFKuGKhHtFQvITqsfk3_FsJDT-8tlwxuN5TwOnbk8q5f0kA%3D%3D&tracking_referrer=www.washingtonpost.com
      
    """
    In particular, the Gaia DR2 solution shows exceptionally large covariances, suggesting that it is unwise to simply interpret the astrometry as an accurate parallax measurement (seeMethods)
    """

        
    # ra, dec = (92.95448746, 22.82574178)
    # Gaia DR2 3425096028968232832
    # astrometric jitter = 1.45
    gaia_ruwe = 1.45116

    # astrometric_n_bad_obs_al = 3
    # astrometric_n_obs_al = 103
    # astrometric_n_obs_ac = 103
    # astrometric_n_good_obs_al = 100

    astrometric_n_obs = 100


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



    # Assume observed at uniformly random times.
    t = obs_start + np.linspace(0, 1, astrometric_n_obs) * (obs_end - obs_start)

    kwds.update(t=t)

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
    - distance (2.14 - 3 * 0.35, 4.23 + 0.24 * 3)
    """

    processes = 12
    n_repeats = 1

    #0.77,
    #5.37
    i_bins = np.linspace(0, 90, 10) # u.deg
    
    n_bins = 25
    m1_bins = np.linspace(0.70, 70, n_bins) # u.solMass
    m2_bins = np.linspace(1, 10, n_bins) # u.solMass
    distance_bins = np.linspace(2.14 - 5 * 0.35,
                                4.23 + 5 * 0.24,
                                n_bins) # u.kpc

    pool = mp.Pool(processes=processes)


    results = []

    grid = np.array(list(itertools.product(i_bins, m1_bins, m2_bins, distance_bins)))

    grid_ruwe = np.zeros((grid.shape[0], n_repeats))
    grid_aen = np.zeros((grid.shape[0], n_repeats))


    for j, (i, m1, m2, distance) in enumerate(tqdm(grid)):

        o_kw = origin_kwds.copy()
        o_kw.update(distance=distance * u.kpc)

        origin = coord.ICRS(**o_kw)
        kwds.update(i=i * u.deg,
                    m1=m1 * u.solMass,
                    m2=m2 * u.solMass,
                    origin=origin,
                    distance=origin.distance)

        for k in range(n_repeats):    
            # Time realisation
            kwds.update(t=obs_start + np.linspace(0, 1, astrometric_n_obs) * (obs_end - obs_start))

            if processes == 1:
                grid_ruwe[j, k], _ = approximate_ruwe(**kwds)
                grid_aen[j, k], _ = astrometric_excess_noise(**kwds)

            else:
                results.append(pool.apply_async(worker, (j, k, kwds.copy())))


    if processes > 1:

        for each in tqdm(results):
            j, k, ruwe, aen = each.get(timeout=1)
            grid_ruwe[j, k] = ruwe
            grid_aen[j, k] = aen


        pool.close()
        pool.join()

    # Take mean from repeats?
    # TODO:
    ruwe_tolerance = 0.1
    mask = np.abs(gaia_ruwe - grid_ruwe) < ruwe_tolerance

    X = grid[mask]

    fig = corner(X, labels=(r"$i$", r"$M_1$", r"$M_2$", r"$d$"))
    fig.savefig("abc.png", dpi=300)
        

    raise a



