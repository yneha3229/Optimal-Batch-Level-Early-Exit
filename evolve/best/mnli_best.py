import numpy as np

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
    rho_small = a / (b * n_active) < 1

    # Base thresholds based on layers left and cost regime
    if layers_left <= 2:
        base_threshold = 0.95
    elif layers_left <= 4:
        base_threshold = 0.93
    elif layers_left <= 6:
        base_threshold = 0.91
    elif layers_left <= 8:
        base_threshold = 0.89
    elif layers_left <= 10:
        base_threshold = 0.87
    else:
        base_threshold = 0.85

    # Adjust the threshold based on cost regime for efficiency
    if rho_small < 1.2:
        adjustment_cost = -0.06 * (exit_idx + 1)
    elif rho_small < 1.4:
        adjustment_cost = -0.04 * (exit_idx + 1)
    elif rho_small < 1.6:
        adjustment_cost = -0.03 * (exit_idx + 1)
    else:
        adjustment_cost = 0

    # Adjust the threshold based on number of active inputs for fine-grained control
    adjustment_active = min(0, -0.02 * ((batch_size - n_active) / batch_size))

    # Calculate the final threshold
    threshold_exit = max(0.9, base_threshold + adjustment_cost + adjustment_active)

    exit_mask = confs >= threshold_exit

    # Determine if we should stop the batch based on layers left and cost ratio
    stop_batch = False
    if layers_left <= (3 + exit_idx):
        stop_batch = True
    elif rho_small < 1.5 and n_active < batch_size // 4:
        stop_batch = True

    return exit_mask, stop_batch