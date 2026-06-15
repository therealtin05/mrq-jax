"""Modified from https://github.com/instadeepai/flashbax/blob/main/flashbax/buffers/trajectory_buffer.py
Changes made:
 - Current_index is now a vector of shape (add_batch_size,)
 - Add a mask argument to the add function 
 - Searchsorted based prioritized sampling)
"""

import functools
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar

from requests import head

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array

from MRQ.buffers.trajectory_buffer import TrajectoryBufferSample
from flashbax import utils

Experience = TypeVar("Experience", bound=chex.ArrayTree)
Priorities = Array  # p in the PER paper
Probabilities = Array  # P in the PER paper
Indices = Array

@dataclass(frozen=True)
class PrioritisedTrajectoryBufferState(Generic[Experience]):
    """State of the prioritised trajectory replay buffer.

    Attributes:
        experience: Arbitrary pytree containing the experience data, for example a single
            timestep (s,a,r). These are stacked along the first axis.
        priority: Array of priorities for each experience in the buffer.
        current_index: Index where the next batch of experience data will be added to.
        is_full: Whether the buffer state is completely full with experience (otherwise it will
            have some empty padded values).
        max_priority: The maximum priority of experience in the buffer. New transitions are added 
            with this magnitude.
    """

    experience: Experience
    priority: Array
    current_index: Array
    is_full: Array
    max_priority: Array


@dataclass(frozen=True)
class PrioritisedTrajectoryBufferSample(TrajectoryBufferSample, Generic[Experience]):
    """Container for samples from the prioritised buffer.

    Attributes:
        indices: Indices corresponding to the sampled experience.
        probabilities: probabilities of the sampled experience.
    """

    indices: Indices
    probabilities: Probabilities
    

def init(
    experience: Experience,
    add_batch_size: int,
    max_length_time_axis: int,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """
    Initialise the buffer state.

    Args:
        experience: A single timestep (e.g. (s,a,r)) used for inferring
            the structure of the experience data that will be saved in the buffer state.
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
            function. I.e. the leading batch size of added experience should have size
            `add_batch_size`.
        max_length_time_axis: Maximum length of the buffer along the time axis (second axis of the
            experience data).

    Returns:
        state: Initial state of the replay buffer. All values are empty as no experience has
            been added yet.
    """
    # Set experience value to be empty.
    experience = jax.tree.map(jnp.empty_like, experience)
    priority = jnp.zeros((add_batch_size, max_length_time_axis), dtype=jnp.float32)
    # Broadcast to [add_batch_size, max_length_time_axis]
    experience = jax.tree.map(
        lambda x: jnp.broadcast_to(
            x[None, None, ...], (add_batch_size, max_length_time_axis, *x.shape)
        ),
        experience,
    )

    return PrioritisedTrajectoryBufferState(
        experience=experience,
        priority=priority,
        is_full=jnp.zeros(add_batch_size, dtype=bool),
        current_index=jnp.zeros(add_batch_size, dtype=int),
        max_priority=jnp.array([1.0], dtype=jnp.float32),
    )


def add(
    state: PrioritisedTrajectoryBufferState[Experience],
    batch: Experience,
    mask: chex.Array | None = None,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """
    Add a batch of experience to the buffer state. Assumes that this carries on from the episode
    where the previous added batch of experience ended. For example, if we consider a single
    trajectory within the batch; if the last timestep of the previous added trajectory's was at
    time `t` then the first timestep of the current trajectory will be at time `t + 1`.

    Args:
        state: The buffer state.
        batch: A batch of experience. The leading axis of the pytree is the batch dimension.
            This must match `add_batch_size` and the structure of the experience used
            during initialisation of the buffer state. This batch is added along the time axis of
            the buffer state.
        mask: A boolean array of shape (add_batch_size,) which specifies which rows of the batch

    Returns:
        A new buffer state with the batch of experience added.
    """

    if mask is None:
        mask = jnp.ones(state.current_index.shape, dtype=bool) # (add_batch_size,)


    # Check that the batch has the correct shape and dtypes.
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    # Get the length of the time axis of the buffer state.
    max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)[1]
    # Check that the sequence length is less than or equal the maximum length of the time axis.
    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )
    # Determine how many timesteps are in this batch.
    seq_len = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    add_batch_size = utils.get_tree_shape_prefix(batch, n_axes=2)[0]
    # Compute the time indices where the new data will be written.
    indices = (jnp.arange(seq_len)[None, :] + state.current_index[:, None]) % max_length_time_axis # (add_batch_size, seq_len)
    indices = jnp.where(mask[:, None], indices, max_length_time_axis) # (add_batch_size, seq_len) utilizing "promise_in_bound"
    env_idx = jnp.arange(add_batch_size)[:, None] # (add_batch_size, 1)

    # Update the buffer state.
    new_experience = jax.tree.map(
        lambda exp_field, batch_field: exp_field.at[env_idx, indices].set(batch_field),
        state.experience,
        batch,
    )

    new_priority = state.priority.at[env_idx, indices].set(state.max_priority[:, None])

    new_current_index = jnp.where(mask, state.current_index + seq_len, state.current_index)
    new_is_full = jnp.where(state.is_full | (new_current_index >= max_length_time_axis), True, False)
    new_current_index = new_current_index % max_length_time_axis

    return state.replace(  # type: ignore
        experience=new_experience,
        current_index=new_current_index,
        is_full=new_is_full,
        priority=new_priority,
    )


def sample(
    state: PrioritisedTrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sequence_length: int,
    period: int,
    priority_exponent: float,
) -> PrioritisedTrajectoryBufferSample[Experience]:
    """
    Sample a batch of trajectories from the buffer.

    Args:
        state: The buffer's state.
        rng_key: Random key.
        batch_size: Batch size of sampled experience.
        sequence_length: Length of trajectory to sample.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.

    Returns:
        A batch of experience.
    """

    assert period == 1, "Only period=1 is currently supported."

    # Get add_batch_size and the full size of the time axis.
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )

    max_start = state.current_index - sequence_length # (add_batch_size,) max_start might be negative 

    valid_item_mask = jnp.where(
        state.is_full[:, None],
        jnp.ones((add_batch_size, max_length_time_axis), dtype=bool).at[jnp.arange(add_batch_size)[:, None], jnp.arange(sequence_length)[None, :] + max_start[:, None]].set(False),
        jnp.arange(max_length_time_axis)[None, :] < max_start[:, None]
    )

    priority = jnp.where(
        valid_item_mask, state.priority, 0.0
    )

    print("flattened priority", priority.flatten())
    flatten_priority = priority.flatten() ** priority_exponent  # (add_batch_size * max_length_time_axis,)
    csum = jnp.cumsum(flatten_priority) # (add_batch_size * max_length_time_axis,)
    total_priority = csum[-1] # scalar

    sampled_priority = jax.random.uniform(rng_key, (batch_size,), minval=0, maxval=total_priority) # (batch_size,)
    sampled_indices = jnp.searchsorted(csum, sampled_priority) # (batch_size,)

    row_idx = sampled_indices // max_length_time_axis # (batch_size,)
    col_idx = sampled_indices % max_length_time_axis # (batch_size,)

    # Create indices for the full subsequence.
    traj_time_indices = (
        col_idx[:, None] + jnp.arange(sequence_length)
    ) % max_length_time_axis  # (batch_size, sequence_length)


    batch_trajectory = jax.tree.map(
        lambda x: x[row_idx[:, None], traj_time_indices], # (batch_size, sequence_length, *x.shape[2:])
        state.experience,
    ) # (batch_size, sequence_length, *experience_shape)


    probabilities = flatten_priority[sampled_indices] / total_priority # (batch_size, sequence_length)

    return PrioritisedTrajectoryBufferSample(
        experience=batch_trajectory,
        indices=sampled_indices,
        probabilities=probabilities
    )


def can_sample(
    state: PrioritisedTrajectoryBufferState[Experience], min_length_time_axis: int
) -> Array:
    """Indicates whether the buffer has been filled above the minimum length, such that it
    may be sampled from."""
    return jnp.all(state.is_full | (state.current_index >= min_length_time_axis))


BufferState = TypeVar("BufferState", bound=PrioritisedTrajectoryBufferState)
BufferSample = TypeVar("BufferSample", bound=PrioritisedTrajectoryBufferSample)


def set_priorities(
    state: BufferState,
    indices: Indices,
    priorities: Priorities,
) -> BufferState:
    """Set the priorities of experience in the buffer state at the specified indices."""
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )
    row_idx = indices // max_length_time_axis # (batch_size,)
    col_idx = indices % max_length_time_axis # (batch_size,)

    new_priority = state.priority.at[row_idx, col_idx].set(priorities)

    new_max_priority = jnp.maximum(state.max_priority, jnp.max(priorities))

    return state.replace(  # type: ignore
        priority=new_priority,
        max_priority=new_max_priority,
    )

@dataclass(frozen=True)
class PrioritisedTrajectoryBuffer(Generic[Experience, BufferState, BufferSample]):
    """Pure functions defining the trajectory buffer. This buffer assumes batches added to the
    buffer are a pytree with a shape prefix of (batch_size, trajectory_length). Consecutive batches
    are then concatenated along the second axis (i.e. the time axis). During sampling this allows
    for trajectories to be sampled - by slicing consecutive sequences along the time axis.

    Attributes:
        init: A pure function which may be used to initialise the buffer state using a single
            timestep (e.g. (s,a,r)).
        add: A pure function for adding a new batch of experience to the buffer state.
        sample: A pure function for sampling a batch of data from the replay buffer, with a leading
            axis of size (`sample_batch_size`, `sample_sequence_length`). Note `sample_batch_size`
            and `sample_sequence_length` may be different to the batch size and sequence length of
            data added to the state using the `add` function.
        can_sample: Whether the buffer can be sampled from, which is determined by if the
            number of trajectories added to the buffer state is greater than or equal to the
            `min_length`.

    See `make_trajectory_buffer` for how this container is instantiated.
    """

    init: Callable[[Experience], BufferState]
    add: Callable[
        [BufferState, Experience],
        BufferState,
    ]
    sample: Callable[
        [BufferState, chex.PRNGKey],
        BufferSample,
    ]
    can_sample: Callable[[BufferState], Array]
    set_priorities: Callable[[BufferState, Indices, Priorities], BufferState]


def validate_size(
    max_length_time_axis: Optional[int], max_size: Optional[int], add_batch_size: int
) -> None:
    if max_size is not None and max_length_time_axis is not None:
        raise ValueError(
            "Cannot specify both `max_size` and `max_length_time_axis` arguments."
        )
    if max_size is not None:
        warnings.warn(
            "Setting max_size dynamically sets the `max_length_time_axis` to "
            f"be `max_size`//`add_batch_size = {max_size // add_batch_size}`."
            "This allows one to control exactly how many timesteps are stored in the buffer."
            "Note that this overrides the `max_length_time_axis` argument.",
            stacklevel=1,
        )


def validate_trajectory_buffer_args(
    max_length_time_axis: Optional[int],
    min_length_time_axis: int,
    add_batch_size: int,
    sample_sequence_length: int,
    period: int,
    max_size: Optional[int],
) -> None:
    """Validate the arguments of the trajectory buffer."""

    validate_size(max_length_time_axis, max_size, add_batch_size)

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    if sample_sequence_length > min_length_time_axis:
        warnings.warn(
            "`sample_sequence_length` greater than `min_length_time_axis`, therefore "
            "overriding `min_length_time_axis`"
            "to be set to `sample_sequence_length`, as we need at least `sample_sequence_length` "
            "timesteps added to the buffer before we can sample.",
            stacklevel=1,
        )
        min_length_time_axis = sample_sequence_length

    if period > sample_sequence_length:
        warnings.warn(
            "Setting period greater than sample_sequence_length will result in no overlap between"
            f"trajectories, however, {period-sample_sequence_length} transitions will "
            "never be sampled. Setting period to be equal to sample_sequence_length will "
            "also result in no overlap between trajectories, however, all transitions will "
            "be sampled. Setting period to be `sample_sequence_length - 1` is generally "
            "desired to ensure that only starting and ending transitions are shared "
            "between trajectories allowing for utilising last transitions for bootstrapping.",
            stacklevel=1,
        )

    if max_length_time_axis is not None:
        if sample_sequence_length > max_length_time_axis:
            raise ValueError(
                "`sample_sequence_length` must be less than or equal to `max_length_time_axis`."
            )

        if min_length_time_axis > max_length_time_axis:
            raise ValueError(
                "`min_length_time_axis` must be less than or equal to `max_length_time_axis`."
            )


def make_prioritised_trajectory_buffer(
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    period: int,
    min_length_time_axis: int,
    max_size: Optional[int] = None,
    max_length_time_axis: Optional[int] = None,
    priority_exponent: float = 0.6,
) -> PrioritisedTrajectoryBuffer:
    """Makes a prioritised trajectory buffer.

    Args:
        add_batch_size: Batch size of experience added to the buffer. Used to initialise the leading
            axis of the buffer state's experience.
        sample_batch_size: Batch size of experience returned from the `sample` method of the
            buffer.
        sample_sequence_length: Trajectory length of experience of sampled batches. Note that this
            may differ from the trajectory length of experience added to the buffer.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        min_length_time_axis: Minimum length of the buffer (along the time axis) before sampling is
            allowed.
        max_size: Optional argument to specify the size of the buffer based on timesteps.
            This sets the maximum number of timesteps that can be stored in the buffer and sets
            the `max_length_time_axis` to be `max_size`//`add_batch_size`. This allows one to
            control exactly how many timesteps are stored in the buffer. Note that this
            overrides the `max_length_time_axis` argument.
        max_length_time_axis: Optional Argument to specify the maximum length of the buffer in terms
            of time steps within the 'time axis'. The second axis (the time axis) of the buffer
            state's experience field will be of size `max_length_time_axis`.


    Returns:
        A trajectory buffer.
    """
    validate_trajectory_buffer_args(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=min_length_time_axis,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=period,
        max_size=max_size,
    )

    if sample_sequence_length > min_length_time_axis:
        min_length_time_axis = sample_sequence_length

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    assert max_length_time_axis is not None
    init_fn = functools.partial(
        init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
    )
    add_fn = functools.partial(
        add,
    )
    sample_fn = functools.partial(
        sample,
        batch_size=sample_batch_size,
        sequence_length=sample_sequence_length,
        period=period,
        priority_exponent=priority_exponent,
    )
    can_sample_fn = functools.partial(
        can_sample, min_length_time_axis=min_length_time_axis
    )
    set_priorities_fn = functools.partial(
        set_priorities,
    )

    return PrioritisedTrajectoryBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        set_priorities=set_priorities_fn,
    )