# expr: ('-', ('/', ('max', ('max', ('min', ('-', 'ok', 'step'), ('min', 'ci', 'odeg')), ('min', 'delta', 'ci')), 'temp'), ('min', ('-', 'ik', 'co'), 'co')), 'delta')
import math


def score(f):
    delta, ci, co, ik, ok, ideg, odeg, kept, n, temp, step = f
    return ((max(max(min((ok - step), min(ci, odeg)), min(delta, ci)), temp) / (abs(min((ik - co), co)) + 1.0)) - delta)
