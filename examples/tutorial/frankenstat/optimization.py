#!/usr/bin/env python
"""SVI optimization utilities for pulsar timing array analysis."""

from functools import partial

import jax
import numpy as np
import numpyro
import optax
from loguru import logger
from numpyro.infer import SVI, Trace_ELBO


def setup_svi(
    model,
    guide,
    loss=None,
    num_warmup_steps=500,
    max_epochs=5000,
    peak_lr=0.01,
    gradient_clipping_val: float | None = None,
):
    """Set up Stochastic Variational Inference with warmup and cosine decay.

    Configures an SVI optimizer with AdamW and a learning rate schedule that
    includes warmup followed by cosine decay. Optionally includes gradient clipping.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    guide : callable
        NumPyro guide function (e.g., AutoDelta, AutoNormal).
    loss : numpyro.infer.ELBO, optional
        Loss function for SVI, by default Trace_ELBO().
    num_warmup_steps : int, optional
        Number of warmup steps for learning rate schedule, by default 500.
    max_epochs : int, optional
        Maximum number of training epochs for decay schedule, by default 5000.
    peak_lr : float, optional
        Peak learning rate after warmup, by default 0.01.
    gradient_clipping_val : float or None, optional
        Maximum gradient norm for clipping. If None, no clipping is applied,
        by default None.

    Returns
    -------
    numpyro.infer.SVI
        Configured SVI object ready for training.

    """
    if loss is None:
        loss = Trace_ELBO()
    # Define the learning rate schedule
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=peak_lr,
        warmup_steps=num_warmup_steps,
        decay_steps=max_epochs,
        end_value=peak_lr * 0.01,  # Decay to 10% of the peak
    )
    npyro_optimizer = numpyro.optim.optax_to_numpyro(
        # Gradient clipping if supplied
        (
            optax.adamw(learning_rate=learning_rate_schedule)
            if gradient_clipping_val is None
            else optax.chain(
                optax.clip_by_global_norm(
                    gradient_clipping_val,
                ),
                optax.adamw(learning_rate=learning_rate_schedule),
            )
        ),
    )
    return SVI(model, guide, npyro_optimizer, loss=loss)


@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch(svi, svi_state, rng_key, batch_size):
    """Run SVI updates for a fixed number of steps using JAX scan.

    This JIT-compiled function efficiently runs multiple SVI update steps
    using `jax.lax.scan` for improved performance.

    Parameters
    ----------
    svi : numpyro.infer.SVI
        SVI object configured with model, guide, and optimizer.
    svi_state : SVIState
        Current state of the SVI optimizer.
    rng_key : jax.random.PRNGKey
        Random number generator key.
    batch_size : int
        Number of SVI update steps to perform.

    Returns
    -------
    final_svi_state : SVIState
        Updated SVI state after batch_size steps.
    final_rng_key : jax.random.PRNGKey
        Updated random key.

    """

    def body_fn(carry, x):
        svi_state, rng_key = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        return (new_svi_state, subkey), None

    # Use lax.scan to loop `body_fn` for `BATCH_SIZE` iterations
    (final_svi_state, final_rng_key), _ = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state, final_rng_key


def run_svi_early_stopping(
    rng_key,
    svi,
    batch_size=500,
    patience=3,
    max_num_batches=20,
):
    """Run SVI training with early stopping based on validation loss.

    Trains the SVI model in batches, monitoring the validation loss and stopping
    early if the loss does not improve for a specified number of consecutive batches.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        Random number generator key for initialization.
    svi : numpyro.infer.SVI
        Configured SVI object with model, guide, and optimizer.
    batch_size : int, optional
        Number of SVI steps per batch, by default 500.
    patience : int, optional
        Number of batches without improvement before early stopping, by default 3.
    max_num_batches : int, optional
        Maximum number of batches to train, by default 20.

    Returns
    -------
    dict
        Dictionary of optimized parameters from the best SVI state.

    """
    svi_state = svi.init(rng_key)

    best_val_loss = float("inf")
    best_svi_state = svi_state
    patience_counter = 0

    logger.info(f"Starting training with batches of {batch_size} steps.")

    final_params = None
    for batch_num in range(max_num_batches):
        svi_state, rng_key = run_training_batch(svi, svi_state, rng_key, batch_size)
        current_val_loss = svi.evaluate(svi_state)
        total_steps = (batch_num + 1) * batch_size
        logger.info(
            f"Batch {batch_num + 1}/{max_num_batches} | Total Steps Possible: {total_steps}",
        )

        # Early stopping logic
        logger.debug(f"{current_val_loss=}")
        logger.debug(f"{best_val_loss=}")
        ratio = current_val_loss - best_val_loss if batch_num >= 1 else -np.inf
        if ratio < -1:
            logger.info(
                f"Loss improved from {best_val_loss:.4f} to {current_val_loss:.4f} {ratio=}. Saving state.",
            )
            best_val_loss = current_val_loss
            best_svi_state = svi_state
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Loss did not improve. Patience: {patience_counter}/{patience} {ratio=}",
            )

            if patience_counter >= patience:
                logger.info("Early stopping triggered. Halting training.")
                break

            logger.info("Optimization complete.")
            logger.info(f"Best loss achieved: {best_val_loss:.4f}")

            final_params = svi.get_params(best_svi_state)

    # This conditional is entered if we exhaust the max training batches
    # without early stopping
    if final_params is None:
        final_params = svi.get_params(best_svi_state)

    return final_params
