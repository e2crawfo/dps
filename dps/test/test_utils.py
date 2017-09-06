import numpy as np
import matplotlib.pyplot as plt

from dps.utils import (
    Polynomial, Poly, Exponential, Exp, Reciprocal, Constant,
    MixtureSchedule, ChainSchedule, RepeatSchedule
)


def test_schedule():
    components = "[Exp(2.0, 0.9, 100, end=1.0), Constant(2), Poly(2.0, 5000, end=1.0), Reciprocal(2.0, 5000, end=1.0)]"

    schedules = [
        "MixtureSchedule({}, 115, shared_clock=True, p=[0.1, 0.2, 0.3, 0.4])".format(components),
        "MixtureSchedule({}, 115, shared_clock=False, p=[0.1, 0.2, 0.3, 0.4])".format(components),
        "MixtureSchedule({}, 115, shared_clock=True, p=None)".format(components),
        "MixtureSchedule({}, 115, shared_clock=False, p=None)".format(components),
        "ChainSchedule({}, 1050, shared_clock=True)".format(components),
        "ChainSchedule({}, 1050, shared_clock=False)".format(components),
        "RepeatSchedule(Exp(2.0, 0.9, 100, end=1.0), 1000)",
        "Exp(2.0, 0.9, 100, end=1.0)",
        "Poly(2.0, 5000, end=1.0)",
        "Reciprocal(2.0, 5000, end=1.0)",
    ]
    for schedule in schedules:
        s = eval(schedule)
        signal = s.build(np.arange(10000).astype('f'))
        plt.plot(signal, label=schedule)
    plt.legend()
    plt.show()
