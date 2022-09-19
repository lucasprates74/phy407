import numpy as np

def p(x):
    return (1 - x) ** 8


def q(x):
    return 1 - 8 * x + 28 * x ** 2 - 56 * x ** 3 + 70 * x ** 4 - 56 * x ** 5 + 28 * x ** 6 - 8 * x ** 7 + x ** 8