import math


def priority(v, ctx):
    """Vertex to delete next (highest first) when breaking cycles. Classic rule: in-degree x out-degree."""
    return ctx.in_deg(v) * ctx.out_deg(v)
