import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from scipy_toolbox.signal import full_width_at


if __name__ == '__main__':
    x = np.linspace(-4, 4, 250)
    y = norm.pdf(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    colors = iter(['orange', 'green', 'red'])
    for p in [0.25, 0.50, 0.75]:
        width = full_width_at(x, y, p=p)
        ax.hlines(p*y.max(), -width/2, width/2, label=f'p = {p}', color=f'tab:{next(colors)}')
    ax.legend()
    plt.show()
