import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from scipy_toolbox.fitting.gauss import fit_sigma_from_fwhm, gauss


if __name__ == '__main__':
    x = np.linspace(-4, 4, 250)
    y = norm.pdf(x)
    sigma = fit_sigma_from_fwhm(x, y)

    fig, ax = plt.subplots()
    ax.set_title(f'Fitted sigma: {sigma}')
    ax.plot(x, y, label='data', lw=7, alpha=0.3)
    ax.plot(
        x,
        gauss(
            x,
            sigma=sigma.nominal_value,
            amplitude=y.max(),
        ),
        ls='--',
        label='fit'
    )
    ax.legend()
    plt.show()
