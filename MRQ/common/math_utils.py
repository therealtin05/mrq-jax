import jax
import jax.numpy as jnp


def soft_ce(pred, target):
    """
    Soft cross-entropy for two-hot targets.

    Args:
        pred: shape (..., num_bins)
        target: shape (..., num_bins)
    Returns:
        loss: shape (...)
    """
    pred = jax.nn.log_softmax(pred, axis=-1)
    return -jnp.sum(pred * target, axis=-1)


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def symexp_bins(low: float, high: float, num_bins: int) -> jnp.ndarray:
    bins = jnp.linspace(low, high, num_bins)
    return jnp.sign(bins) * (jnp.exp(jnp.abs(bins)) - 1)


def two_hot(
    x: jnp.ndarray,
    low: float,
    high: float,
    num_bins: int,
) -> jnp.ndarray:
    """
    Two-hot encoding using symexp-spaced bins in reward space.
    """

    bins = symexp_bins(low, high, num_bins)
    x_expanded = x[..., None]
    diff = x_expanded - bins
    diff = diff - 1e8 * (jnp.sign(diff) - 1)
    ind = jnp.argmin(diff, axis=-1)

    lower = bins[ind]
    upper_ind = jnp.minimum(ind + 1, num_bins - 1)
    upper = bins[upper_ind]
    weight = (x - lower) / (upper - lower)

    soft_two_hot = (
        jax.nn.one_hot(ind, num_classes=num_bins) * (1 - weight)[..., None]
        + jax.nn.one_hot(upper_ind, num_classes=num_bins) * weight[..., None]
    )
    return soft_two_hot


def two_hot_inv(
    x: jnp.ndarray,
    low: float,
    high: float,
    num_bins: int,
    apply_softmax: bool = True,
) -> jnp.ndarray:
    """
    Decode bin logits to a scalar reward in reward space.
    """
    bins = symexp_bins(low, high, num_bins)
    if apply_softmax:
        x = jax.nn.softmax(x, axis=-1)
    return jnp.sum(x * bins, axis=-1)
