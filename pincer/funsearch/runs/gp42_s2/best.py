# expr: ('+', ('*', ('-', 'i', ('/', 'oi', ('+', 'oi', ('sqrt', ('*', 'i', ('min', 'a', 'm')))))), 'o'), ('*', 'a', ('min', 'n', ('*', 'n', ('log', 'o')))))
import math


def priority(v, ctx):
    al = ctx.alive
    i = ctx.in_deg(v)
    o = ctx.out_deg(v)
    m = ctx.mutual_deg(v)
    ki = (ctx.inn[v] & ctx.kept).bit_count()
    ko = (ctx.out[v] & ctx.kept).bit_count()
    a = al.bit_count()
    s = ctx.step
    n = ctx.n
    oi = 0
    x = ctx.out[v] & al
    while x:
        low = x & -x
        oi += ctx.in_deg(low.bit_length() - 1)
        x ^= low
    io = 0
    x = ctx.inn[v] & al
    while x:
        low = x & -x
        io += ctx.out_deg(low.bit_length() - 1)
        x ^= low
    return (((i - (oi / (abs((oi + math.sqrt(abs((i * min(a, m)))))) + 1.0))) * o) + (a * min(n, (n * math.log1p(abs(o))))))
