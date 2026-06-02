from typing import NamedTuple

import jax.numpy as jnp


class RunningPercentileState(NamedTuple):
    """Running statistics for observtations/rewards"""
    value: jnp.ndarray


def update_running_value(
    rp_state: RunningPercentileState, x: jnp.ndarray, tau: float,
) -> RunningPercentileState:
    
    """Percentile scaling, less sensitive to outlier than Standandization.
    Only scale by x / (percentile_95 - percentile_5)
    Args:
        std_state: RunningPercentileState
        x: (batch_size, )
        tau: float
    """

    # Compute percentiles across batch (axis=0)
    percentiles = jnp.percentile(x, jnp.array([5.0, 95.0]), axis=0)
    scale = percentiles[1] - percentiles[0]
    new_value = tau * scale + (1.0 - tau) * rp_state.value
    new_value = jnp.clip(new_value, 1, None)

    return RunningPercentileState(value=new_value)


def normalize_with_rp(
    x: jnp.ndarray,
    rp: RunningPercentileState,
) -> jnp.ndarray:
    """Normalize input with provided running statistics"""

    return x / rp.value

if __name__ == '__main__':
    import jax
    
    key = jax.random.PRNGKey(0)
    dim = 10
    state = RunningPercentileState(value=jnp.ones(dim))
    print(f"Initial value: {state.value}")
    
    # Test with random data
    for i in range(10): 
        key, subkey = jax.random.split(key)
        # x = jax.random.normal(subkey, (100,dim))
        x = jnp.arange(0, 100*dim).reshape(100,dim) # should conversge to same value for all dim)
        state = update_running_value(state, x, 0.01)
        print(f"After update: {state.value}") 