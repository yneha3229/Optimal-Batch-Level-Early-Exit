"""Seed program for FunSearch / OpenEvolve: a plain per-sample confidence threshold (DeeBERT-style).
Only the code between the EVOLVE-BLOCK markers is rewritten."""
import numpy as np


# EVOLVE-BLOCK-START
def decide(exit_idx, confs, n_active, batch_size, layers_left, a, b):
    """Called at every exit point of a batch of `batch_size` inputs, of which `n_active` are still running.

    exit_idx     index of this exit (0 = first; exits sit at layers 2, 4, 6, 8, 10, 12)
    confs        np.ndarray [n_active], max softmax probability of each active input at this exit
    layers_left  transformer layers still to run if nobody stops
    a, b         measured cost of one layer on n active inputs: a + b * n milliseconds

    Return (exit_mask, stop_batch):
      exit_mask   bool array [n_active]; True = this input leaves now with its current prediction
      stop_batch  True = every remaining input leaves now too (ends the batch)
    """
    exit_mask = confs > 0.9
    stop_batch = False
    return exit_mask, stop_batch
# EVOLVE-BLOCK-END
