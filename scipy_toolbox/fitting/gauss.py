from collections import namedtuple
from dataclasses import dataclass
from numbers import Number

import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

from ..signal import full_width_at


def fit(x, y, **kwargs):
    """Wrapper around `scipy.optimize.curve_fit` for fitting a Gaussian.

    The following parameters are included:

    * Sigma
    * Amplitude
    * Center
    * Offset

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.
    kwargs
        Remaining arguments for `curve_fit`.

    Returns
    -------
    result : :class:`FitResult`
    """
    popt, pcov = curve_fit(
        gauss, x, y,
        p0=compute_p0_for_gauss(x, y),
        **kwargs,
    )
    perr = np.sqrt(np.diag(pcov))
    return FitResult(*(
        ufloat(val, err)
        for val, err in zip(popt, perr)
    ))


def fit_peak(x, y, *, dx, **kwargs):
    """Wrapper around `scipy.optimize.curve_fit` for fitting the peak region of a Gaussian.

    The following parameters are included:

    * Sigma
    * Amplitude
    * Center

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.
    dx : float or 2-tuple of float
        The x-margin around the peak.
        Either symmetric (float) or asymmetric (2-tuple).
    kwargs
        Remaining arguments for `curve_fit`.

    Returns
    -------
    result : :class:`FitResult`
        The `offset` is not included in the fit and hence its standard deviation
        is set to zero.
    """
    if isinstance(dx, Number):
        dx = (dx, dx)

    offset = compute_p0_offset(x, y)
    y = y - offset

    amplitude = compute_p0_amplitude(x, y, k=3)
    center = compute_p0_center(x, y, k=3)
    il = np.searchsorted(x, center - 0.9*dx[0])  # 10% margin to +-dx fit region
    ir = np.searchsorted(x, center + 0.9*dx[1])
    sigma = np.mean([
        dx[0] / np.sqrt(2*np.log(amplitude/y[il])),
        dx[1] / np.sqrt(2*np.log(amplitude/y[ir])),
    ])

    mask = np.logical_and(
        x >= center - dx[0],
        x <= center + dx[1],
    )
    popt, pcov = curve_fit(
        gauss, x[mask], y[mask],
        p0=(sigma, amplitude, center),
        **kwargs,
    )
    perr = np.sqrt(np.diag(pcov))
    popt = [*popt, offset]  # add constant offset
    perr = [*perr, 0]
    return FitResult(*(
        ufloat(val, err)
        for val, err in zip(popt, perr)
    ))


def fit_sigma_from_fwhm(x, y, *, p=0.5, k=3):
    """Fit standard deviation of Gaussian from full width at p-th maximum.

    Computes the full width values via :func:`full_width_at` and transforms
    them to standard deviation via the following relationship:

    .. math:: \\sigma = \\frac{fwhm}{2\\sqrt{2\\log p^{-1}}}

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.
    p : float
        The fraction of the maximum at which to compute the full width.
    k : int
        The width is computed separately for each of the top-k values in y.

    Returns
    -------
    value : float
        The average of the top-k widths, converted to sigma.
    std_dev : float
        The standard error of the top-k widths, converted to sigma.
    """
    width_values = full_width_at(x, y, p=p, k=k)
    width = ufloat(np.mean(width_values), np.std(width_values)/np.sqrt(len(width_values)))
    sigma = 0.5*width / np.sqrt(2*np.log(1/p))
    return sigma


@dataclass
class FitResult:
    sigma : ufloat
    amplitude : ufloat
    center : ufloat
    offset : ufloat


P0 = namedtuple('P0', 'sigma amplitude center offset')


def compute_p0_for_gauss(x, y, *, p=0.5):
    """Compute initial parameter guess for :func:`gauss` function.

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.
    p : float
        Fraction of the y-maximum at which to compute the full width.
        This is used to estimate the value of `sigma`.

    Returns
    -------
    initial_guess : :class:`P0`
    """
    return P0(
        compute_p0_sigma(x, y, p=p),
        compute_p0_amplitude(x, y),
        compute_p0_center(x, y),
        compute_p0_offset(x, y),
    )


def compute_p0_sigma(x, y, *, p=0.5):
    """Estimate the sigma from the full width at the p-th fraction of the y-maximum."""
    return 0.5*full_width_at(x, y, p=p) / np.sqrt(2*np.log(1/p))


def compute_p0_amplitude(x, y, *, k=1, k_offset=1):
    """Estimate the amplitude by averaging the top-k y-values (subtracted by the offset)."""
    return np.sort(y)[-k:].mean() - compute_p0_offset(x, y, k=k_offset)


def compute_p0_center(x, y, *, k=1):
    """Estimate the center from the mean position of the top-k y-values."""
    return x[np.argsort(y)[-k:]].mean()


def compute_p0_offset(x, y, *, k=1):
    """Estimate the offset from the k left- and rightmost y-values."""
    return y[[*range(0, k), *range(-k, 0)]].mean()


def gauss(x, sigma, amplitude=1, center=0, offset=0):
    """Gaussian distribution"""
    return amplitude * np.exp(-0.5 * (x - center)**2 / sigma**2) + offset
