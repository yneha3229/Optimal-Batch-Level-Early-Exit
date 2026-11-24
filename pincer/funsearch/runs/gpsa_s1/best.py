# expr: ('-', ('-', ('sq', 0.5), ('sq', 'temp')), ('/', 'delta', ('-', ('sq', 0.5), ('max', ('*', 'n', ('sqrt', 'co')), 'ci'))))
import math


def score(f):
    delta, ci, co, ik, ok, ideg, odeg, kept, n, temp, step = f
    return (((0.5) ** 2 - (temp) ** 2) - (delta / (abs(((0.5) ** 2 - max((n * math.sqrt(abs(co))), ci))) + 1.0)))
