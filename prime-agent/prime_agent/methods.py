"""Existing primality-proving methods, each run in a forked process with a hard time limit.

  pari_isprime   PARI isprime(n): BPSW, then a proof (APR-CL or ECPP, PARI's own choice)
  pari_aprcl     PARI isprime(n, 2): APR-CL proof
  pari_pl        PARI isprime(n, 1): Pocklington-Lehmer (n - 1) proof
  pari_ecpp      PARI primecert(n): ECPP certificate, then primecertisvalid (both timed; certificate checkable)
  flint          FLINT fmpz.is_prime(): BPSW, then n-1/n+1 (BLS) or APR-CL
"""
from __future__ import annotations

import multiprocessing as mp
import time

_PARI = None


def pari():
    global _PARI
    if _PARI is None:
        import cypari2

        _PARI = cypari2.Pari()
        _PARI.allocatemem(1 << 30, silent=True)
    return _PARI


def _pari_isprime(n):
    return bool(pari().isprime(n)), {}


def _pari_aprcl(n):
    return bool(pari().isprime(n, 2)), {}


def _pari_pl(n):
    return bool(pari().isprime(n, 1)), {}


def _pari_ecpp(n):
    P = pari()
    t = time.perf_counter()
    c = P.primecert(n)
    build = time.perf_counter() - t
    ok = bool(P.primecertisvalid(c))
    return ok, {"build_s": round(build, 4), "verify_s": round(time.perf_counter() - t - build, 4)}


def _flint(n):
    import flint

    return bool(flint.fmpz(n).is_prime()), {}


METHODS = {"pari_isprime": _pari_isprime, "pari_aprcl": _pari_aprcl, "pari_pl": _pari_pl, "pari_ecpp": _pari_ecpp,
           "flint": _flint}


def _child(fn, n, conn):
    try:
        t = time.perf_counter()
        ok, info = fn(n)
        conn.send((ok, time.perf_counter() - t, info, ""))
    except Exception as e:  # noqa: BLE001
        conn.send((False, 0.0, {}, repr(e)[:200]))


def run(name, n, limit):
    pari()                      # initialise PARI before forking, so children do not pay for it
    import flint  # noqa: F401

    ctx = mp.get_context("fork")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(METHODS[name], n, b), daemon=True)
    t = time.perf_counter()
    p.start()
    if a.poll(limit):
        ok, s, info, err = a.recv()
    else:
        ok, s, info, err = False, time.perf_counter() - t, {}, "time limit"
    if p.is_alive():
        p.kill()
    p.join()
    return ok, s, info, err
