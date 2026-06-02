"""Defines some types used in QDax"""

from typing import Dict, Generic, TypeVar, Union

from typing import Any

try:
    import brax.envs  # type: ignore
except Exception:  # pragma: no cover
    brax = None  # type: ignore
import jax
import jax.numpy as jnp
from chex import ArrayTree
from typing_extensions import TypeAlias
from flax.struct import PyTreeNode


# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
StateDescriptor: TypeAlias = jnp.ndarray
EnvState: TypeAlias = Any
Params: TypeAlias = ArrayTree

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = ArrayTree
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray
Spread: TypeAlias = jnp.ndarray
Gradient: TypeAlias = jnp.ndarray

Skill: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = Dict[str, ArrayTree]

# Pareto fronts
T = TypeVar("T", bound=Union[Fitness, Genotype, Descriptor, jnp.ndarray])


class ParetoFront(Generic[T]):
    def __init__(self) -> None:
        super().__init__()


Mask: TypeAlias = jnp.ndarray

# Others
RNGKey: TypeAlias = jax.Array
Metrics: TypeAlias = Dict[str, jnp.ndarray]

# TDMPC2
Latent: TypeAlias = jnp.ndarray

class TrainingState(PyTreeNode):
    """The state of a training process. Can be used to store anything
    that is useful for a training process. This object is used in the
    package to store all stateful object necessary for training an agent
    that learns how to act in an MDP.
    """

    pass


