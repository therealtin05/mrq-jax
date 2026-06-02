import jax
import jax.numpy as jnp

from flax import linen as nn
from MRQ.custom_types import Latent, Observation, Action

from typing import Tuple, Callable, Any, Optional
class NormedLinear(nn.Module):
    """Linear layer with LayerNorm, activation.
    """

    hidden_layer_size: int = 256 
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu 
    kernel_init: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    bias: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        hidden = nn.Dense(
            self.hidden_layer_size, use_bias=self.bias, kernel_init=self.kernel_init,
        )(data)
        hidden = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-5)(hidden)
        if self.activation is not None:
            hidden = self.activation(hidden)
        return hidden
    


class MLP(nn.Module):

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    kernel_init: Optional[Callable[..., Any]] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None
    normed_final_layer: bool = False
    @nn.compact
    def __call__(self, data: jnp.ndarray, training: bool=True) -> jnp.ndarray:
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = NormedLinear(
                    hidden_layer_size=hidden_size,
                    kernel_init=self.kernel_init,
                    activation=self.activation,
                    bias=self.bias
                )(hidden, training)

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                if self.normed_final_layer:
                    hidden = NormedLinear(
                        hidden_layer_size=hidden_size,
                        kernel_init=kernel_init,
                        activation=None,
                        bias=self.bias
                    )(hidden, training)
                else:
                    hidden = nn.Dense(
                        hidden_size,
                        kernel_init=kernel_init,
                        use_bias=self.bias,
                    )(hidden)
                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden



class Ensemble(nn.Module):
    num_network: int
    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    kernel_init: Optional[Callable[..., Any]] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")

    @nn.compact
    def __call__(self, data:jnp.ndarray, deterministic:bool=True) -> jnp.ndarray:
        output = []
        for i in range(self.num_network):
            output.append(
                MLP(
                    layer_sizes=self.layer_sizes,
                    activation=self.activation,
                    kernel_init=self.kernel_init,
                    final_activation=self.final_activation,
                    bias=self.bias,
                    kernel_init_final=self.kernel_init_final,
                )(data, deterministic)
            )
        output = jnp.stack(output, axis=-1)  # shape: (batch_size, output_dim, num_network)

        return output
    

class TD3Actor(nn.Module):
    action_size: int
    hidden_layer_size: Tuple[int, ...]
    kernel_init: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    kernel_init_final: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        return MLP(
            layer_sizes=self.hidden_layer_size + (self.action_size,),
            kernel_init=self.kernel_init, 
            kernel_init_final=self.kernel_init_final,
            activation=jax.nn.relu
        )(obs)


class SACActor(nn.Module):
    action_size: int
    hidden_layer_size: Tuple[int, ...]
    kernel_init: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    kernel_init_final: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    @nn.compact
    def __call__(self, obs: Observation) -> jnp.ndarray:
        return MLP(
            layer_sizes=self.hidden_layer_size + (2 * self.action_size,),
            kernel_init=self.kernel_init, 
            kernel_init_final=self.kernel_init_final
        )(obs)


class TD3Critic(nn.Module):
    hidden_layer_size: Tuple[int, ...]
    num_qs: int
    output_dim: int = 1
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    kernel_init: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    kernel_init_final: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    """Critic implemented as an ensemble model

    Attributes:
        hidden_layer_size: the size of the layer in the neural network.
        num_qs: number of Q heads.
        output_dim: number of output dim. Normally it is just 1 as Critic output
                    is in continuous range to work with MSE. However, we are also 
                    interested in working with Soft Cross Entropy loss, or discrete regression,
                    therefore the need to specify a output_dim > 1

    """

    @nn.compact
    def __call__(
        self, 
        obs: Observation, 
        actions: Action, 
        deterministic: bool = True, 
    ) -> jnp.ndarray:
        input_ = jnp.concatenate([obs, actions], axis=-1)

        layer_sizes = self.hidden_layer_size + (self.output_dim,)

        output = Ensemble(
            num_network=self.num_qs,
            layer_sizes=layer_sizes,
            activation=self.activation,
            kernel_init=self.kernel_init,
            kernel_init_final=self.kernel_init_final,
        )(input_, deterministic)

        return output

class MRQCritic(nn.Module):
    hidden_layer_size: Tuple[int, ...]
    num_qs: int
    output_dim: int = 1
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu
    kernel_init: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    kernel_init_final: Callable[..., Any] = nn.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
    """Critic implemented as an ensemble model

    Attributes:
        hidden_layer_size: the size of the layer in the neural network.
        num_qs: number of Q heads.
        output_dim: number of output dim. Normally it is just 1 as Critic output
                    is in continuous range to work with MSE. However, we are also 
                    interested in working with Soft Cross Entropy loss, or discrete regression,
                    therefore the need to specify a output_dim > 1

    """

    @nn.compact
    def __call__(
        self, 
        zsa: Latent, 
        deterministic: bool = True, 
    ) -> jnp.ndarray:

        layer_sizes = self.hidden_layer_size + (self.output_dim,)

        output = Ensemble(
            num_network=self.num_qs,
            layer_sizes=layer_sizes,
            kernel_init=self.kernel_init,
            kernel_init_final=self.kernel_init_final,
            activation=self.activation,
        )(zsa, deterministic)

        return output
