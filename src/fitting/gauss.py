from collections import namedtuple
from dataclass import dataclass

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


def fit_truncated(x, y, **kwargs):
    """Wrapper around `scipy.optimize.curve_fit` for fitting a truncated Gaussian.

    The following parameters are included:

    * Sigma
    * Amplitude
    * Center
    * Offset
    * Left boundary
    * Right boundary

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.
    kwargs
        Remaining arguments for `curve_fit`.

    Returns
    -------
    result : :class:`FitResultTruncatedGauss`
    """
    kwargs.setdefault('method', 'trf')
    p0 = (
        *compute_p0_for_gauss(x, y),
        x[0],
        x[-1],
    )
    popt, pcov = curve_fit(
        truncated_gauss, x, y,
        p0=p0,
        **kwargs,
    )
    perr = np.sqrt(np.diag(pcov))
    return FitResultTruncatedGauss(*(
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
    if isinstance(dx, float):
        dx = (dx, dx)
    p0 = compute_p0_for_gauss(x, y)
    y = y - p0.offset
    x_peak = x[y.argmax()]
    mask = np.logical_and(
        x >= x_peak - dx[0],
        x <= x_peak + dx[1],
    )
    popt, pcov = curve_fit(
        gauss, x, y,
        p0=(p0.sigma, p0.amplitude, p0.center),
        **kwargs,
    )
    perr = np.sqrt(np.diag(pcov))
    popt = [*popt, p0.offset]  # add constant offset
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


@dataclass
class FitResultTruncatedGauss(FitResult):
    left : ufloat
    right : ufloat


P0 = namedtuple('P0', 'sigma amplitude center offset')


def compute_p0_for_gauss(x, y):
    """Compute initial parameter guess for :func:`gauss` function.

    Parameters
    ----------
    x, y : array_like
        x- and y-coordinates of the data.

    Returns
    -------
    initial_guess : :class:`P0`
    """
    return P0(
        0.5*full_width_at(x, y, p=0.5) / np.sqrt(2*np.log(2)),  # sigma
        y.max(),  # amplitude
        x[y.argmax()],  # center
        y[[0, -1]].mean(),  # offset
    )


def gauss(x, sigma, amplitude=1, center=0, offset=0):
    """Gaussian distribution"""
    return amplitude * np.exp(-0.5 * (x - center)**2 / sigma**2) + offset


def truncated_gauss(x, sigma, amplitude=1, center=0, offset=0, left=-np.inf, right=np.inf):
    """Gaussian distribution, truncated at `left` and `right` to value `offset`."""
    result = amplitude * np.exp(-0.5 * (x - center)**2 / sigma**2) + offset
    mask = np.logical_or(
        x < left,
        x > right,
    )
    result[mask] = offset
    return result
