import jax
import jax.numpy as jnp

from typing import Callable, Tuple, Any
from functools import partial

from MRQ.buffers.transitions import Transition
from MRQ.custom_types import TrainingState

from brax.envs import State as EnvState  # type: ignore

from MRQ.custom_types import RNGKey

def multi_step_reward(
    rewards: jnp.ndarray,
    continues: jnp.ndarray,
    discount: float,
) -> jnp.ndarray:
    """
        
    Args:
        rewards: shape (horizon, batch_size,)
        continues: shape (horizon, batch_size,)
        discount: float
    Return:
        discounted_rewards: shape (batch_size)
        final_valid_idx: shape (batch_size, )
        discounts: shape (batch_size, )
    """
    mask = jnp.roll(continues, 1, axis=0)
    mask = mask.at[0].set(1)
    discounts = discount ** jnp.arange(rewards.shape[0]) # shape (horizon, )
    discounted_rewards = jnp.sum(rewards * discounts[..., None] * mask, axis=0) # shape (batch_size) ### discounted reward at each step

    # example 1: continues = [1, 1, 0, 0, 0] then mask = [1, 1, 1, 0, 0] then final_valid_idx = 2
    # example 2: continues = [0, 0, 0, 0, 0] then mask = [1, 0, 0, 0, 0] then final_valid_idx = 0
    # example 3: continues = [1, 1, 1, 1, 1] then mask = [1, 1, 1, 1, 1] then final_valid_idx = -1
    final_valid_idx = jnp.argmin(mask, axis=0) - 1 # shape (batch_size) 
    return discounted_rewards, final_valid_idx, discount * discounts[final_valid_idx] # shape (batch_size), shape (batch_size), shape (batch_size, )



@partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
def generate_unroll(
    init_state: EnvState,
    training_state: TrainingState,
    random_key: RNGKey,
    prev_mean: jnp.ndarray,
    episode_length: int,
    play_step_fn: Callable[
        [EnvState, TrainingState, RNGKey, jnp.ndarray,],
        Tuple[
            EnvState,
            RNGKey,
            jnp.ndarray,
            Transition,
        ],
    ],
) -> Transition:
    """Generates an episode according to the agent's policy, returns the final state of the
    episode and the transitions of the episode.
    """

    def _scan_play_step_fn(
        carry: Tuple[EnvState, TrainingState, RNGKey, jnp.ndarray], unused_arg: Any
    ) -> Tuple[Tuple[EnvState, TrainingState, RNGKey, jnp.ndarray], Transition]:
        env_state, training_state, random_key, prev_mean = carry
        env_state, random_key, cur_mean, transitions = play_step_fn(env_state, training_state, random_key, prev_mean)
        # env_state, random_key, transitions = play_step_fn(env_state, training_state, random_key) ### DEBUG
        return (env_state, training_state, random_key, cur_mean), transitions

    (state, training_state, cur_mean, random_key), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state, training_state, random_key, prev_mean),
        (),
        length=episode_length,
    )
    return transitions


def get_mask_from_transitions(
    data: Transition,
) -> jnp.ndarray:
    """
    Docstring for get_mask_from_transitions
    Args:
        data: shape (env_batch_size, episode_length, ...)
    Return:
        mask: shape # (env_batch_size, episode_length, ...)
    """
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1) # shape (env_batch_size, episode_length, ...)
    mask = jnp.roll(is_done, 1, axis=1) # shape (env_batch_size, episode_length, ...)
    mask = mask.at[:, 0].set(0) # shape (env_batch_size, episode_length, ...)
    return mask