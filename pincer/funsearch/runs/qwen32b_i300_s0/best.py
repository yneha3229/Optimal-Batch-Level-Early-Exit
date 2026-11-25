import math


def priority(v, ctx):
    """Improved version of `priority_v1`."""
    in_deg = ctx.in_deg(v)
    out_deg = ctx.out_deg(v)
    mutual_deg = ctx.mutual_deg(v)
    return mutual_deg + (in_deg + out_deg) * math.log2(1 + min(in_deg, out_deg)) - abs(in_deg - out_deg)
