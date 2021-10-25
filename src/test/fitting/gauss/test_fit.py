import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from scipy_toolbox.fitting.gauss import fit, gauss


if __name__ == '__main__':
    x = np.linspace(-3, 3, 250)
    y = norm.pdf(x)
    result = fit(x, y)

    fig, ax = plt.subplots()
    ax.set_title(f'Fitted sigma: {result.sigma}')
    ax.plot(x, y, label='data')
    ax.plot(
        x,
        gauss(
            x,
            sigma=result.sigma.nominal_value,
            amplitude=result.amplitude.nominal_value,
            center=result.center.nominal_value,
            offset=result.offset.nominal_value,
        ),
        ls='--',
        label='fit'
    )
    ax.legend()
    plt.show()
