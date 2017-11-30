import numpy as np
import matplotlib.pyplot as plt

from dps.utils import (
    Polynomial, Poly, Exponential, Exp, Reciprocal, Constant,
    MixtureSchedule, ChainSchedule, RepeatSchedule
)


def test_schedule(show_plots):
    components = "[Exp(2.0, 1.0, 100, 0.9), Constant(2), Poly(2.0, 1.0, 5000), Reciprocal(2.0, 1.0, 5000)]"

    schedules = [
        "MixtureSchedule({}, 115, shared_clock=True, p=[0.1, 0.2, 0.3, 0.4])".format(components),
        "MixtureSchedule({}, 115, shared_clock=False, p=[0.1, 0.2, 0.3, 0.4])".format(components),
        "MixtureSchedule({}, 115, shared_clock=True, p=None)".format(components),
        "MixtureSchedule({}, 115, shared_clock=False, p=None)".format(components),
        "ChainSchedule({}, 1050, shared_clock=True)".format(components),
        "ChainSchedule({}, 1050, shared_clock=False)".format(components),
        "RepeatSchedule(Exp(2.0, 1.0, 100, 0.9), 1000)",
        "Exp(2.0, 1.0, 100, 0.9)",
        "Poly(2.0, 1.0, 5000)",
        "Reciprocal(2.0, 1.0, 5000)",
    ]
    for schedule in schedules:
        s = eval(schedule)
        signal = s.build(np.arange(10000).astype('f'))
        if show_plots:
            plt.plot(signal, label=schedule)

    if show_plots:
        plt.legend()
        plt.show()
