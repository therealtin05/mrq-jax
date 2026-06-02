from dataclasses import dataclass

import jax
import jax.numpy as jnp

import flax.linen as nn

import optax

from .networks.networks import MLP, TD3Actor, MRQCritic
from .networks.activations import simnorm

from functools import partial

from typing import Tuple, Callable, Optional, List, Dict

from .custom_types import Params, RNGKey, Observation, Latent, Action, Reward, TrainingState
from .common.scale import RunningPercentileState

class WorldModelTrainingState(TrainingState):
    policy_params: Params
    critic_params: Params
    zs_encoder_params: Params
    za_encoder_params: Params
    zsa_encoder_params: Params
    dynamic_params: Params
    reward_params: Params
    termination_params: Params

    target_policy_params: Params
    target_critic_params: Params
    target_zs_encoder_params: Params
    target_za_encoder_params: Params
    target_zsa_encoder_params: Params
    target_dynamic_params: Params
    target_reward_params: Params
    target_termination_params: Params

    policy_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    zs_encoder_optimizer_state: optax.OptState
    zsa_encoder_optimizer_state: optax.OptState
    za_encoder_optimizer_state: optax.OptState
    dynamic_optimizer_state: optax.OptState
    reward_optimizer_state: optax.OptState
    termination_optimizer_state: optax.OptState
    
    rp_state: RunningPercentileState

@dataclass
class WorldModelConfig:
    num_qs: int
    num_bins: int = 1 
    low: int = -10
    high: int = 10
    simnorm_dim: int = 8

    lr: float = 3e-4
    enc_lr: float = 1e-4

    za_encoder_hidden_layer_sizes: Tuple[int, ...] = (256,)
    zs_encoder_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
    zsa_encoder_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
    critic_hidden_layer_sizes: Tuple[int, ...] = (512, 512, 512)
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 512)

    latent_dim: int = 512
    max_grad_norm: float = 30.0 # set to 0.0 means not use gradient clipping

class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig, action_size: int, observation_size: int):
        self._action_size = action_size
        self._observation_size = observation_size
        self._config = config

        self._policy = TD3Actor(
            action_size=action_size,
            hidden_layer_size=self._config.policy_hidden_layer_sizes,
        )
        self._critic = MRQCritic(
            hidden_layer_size=self._config.critic_hidden_layer_sizes,
            num_qs=self._config.num_qs,
            output_dim=1, 
            activation=jax.nn.mish,
            kernel_init_final=nn.initializers.zeros,
        )

        self._dynamic = MLP(
            layer_sizes=(self._config.latent_dim, ),
        )
        self._termination = MLP(
            layer_sizes=(1,),
        )
        self._reward = MLP(
            layer_sizes=(self._config.num_bins,),
            kernel_init_final=nn.initializers.zeros,
        )  
        self._za_encoder = MLP(
            layer_sizes=self._config.za_encoder_hidden_layer_sizes, final_activation=jax.nn.elu,
        )

        self._zs_encoder = MLP(
            layer_sizes=self._config.zs_encoder_hidden_layer_sizes + (self._config.latent_dim,),
            final_activation=jax.nn.elu,
            normed_final_layer=True
        )

        self._zsa_encoder = MLP(
            layer_sizes=self._config.zsa_encoder_hidden_layer_sizes + (self._config.latent_dim,),
        )

        self.policy_optimizer = optax.adamw(learning_rate=self._config.lr)
        self.reward_optimizer = optax.adamw(learning_rate=self._config.enc_lr)
        self.termination_optimizer = optax.adamw(learning_rate=self._config.enc_lr)
        self.dynamic_optimizer = optax.adamw(learning_rate=self._config.enc_lr)
        self.za_encoder_optimizer = optax.adamw(learning_rate=self._config.enc_lr)
        self.zs_encoder_optimizer = optax.adamw(learning_rate=self._config.enc_lr)
        self.zsa_encoder_optimizer = optax.adamw(learning_rate=self._config.enc_lr)

        # prepare optimizer
        if self._config.max_grad_norm > 0:
            grad_clip = optax.clip_by_global_norm(self._config.max_grad_norm)
            self.critic_optimizer = optax.chain(
                grad_clip, optax.adamw(learning_rate=self._config.lr)
            )
        else:
            self.critic_optimizer = optax.adamw(learning_rate=self._config.lr)


    def init(
        self, random_key: RNGKey, 
    ) -> Tuple[WorldModelTrainingState, RNGKey]:

        # define params
        dummy_obs = jnp.zeros((1, self._observation_size))
        dummy_action = jnp.zeros((1, self._action_size))
        dummy_latent = jnp.zeros((1, self._config.latent_dim))
        dummy_zsa_input = jnp.zeros((1, self._config.latent_dim + self._config.za_encoder_hidden_layer_sizes[-1]))

        random_key, policy_key, critic_key, zs_encoder_key, za_encoder_key, zsa_encoder_key, dynamic_key, reward_key, termination_key = \
            jax.random.split(random_key, 9)
        
        # If implement dropout, just put argument trainig=False in the init function
        policy_params = self._policy.init(policy_key, dummy_latent)
        critic_params = self._critic.init(critic_key, dummy_latent)


        zs_encoder_params = self._zs_encoder.init(zs_encoder_key, dummy_obs)
        za_encoder_params = self._za_encoder.init(za_encoder_key, dummy_action)
        zsa_encoder_params = self._zsa_encoder.init(zsa_encoder_key, dummy_zsa_input)

        dynamic_params = self._dynamic.init(dynamic_key, dummy_latent)
        reward_params = self._reward.init(reward_key, dummy_latent)
        termination_params = self._termination.init(termination_key, dummy_latent)


        target_policy_params = jax.tree_util.tree_map(
            lambda x: x, policy_params
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda x: x, critic_params
        )
        target_zs_encoder_params = jax.tree_util.tree_map(
            lambda x: x, zs_encoder_params
        )
        target_za_encoder_params = jax.tree_util.tree_map(
            lambda x: x, za_encoder_params
        )
        target_zsa_encoder_params = jax.tree_util.tree_map(
            lambda x: x, zsa_encoder_params
        )
        target_dynamic_params = jax.tree_util.tree_map(
            lambda x: x, dynamic_params
        )
        target_reward_params = jax.tree_util.tree_map(
            lambda x: x, reward_params
        )
        target_termination_params = jax.tree_util.tree_map(
            lambda x: x, termination_params
        )

        policy_optimzier_state = self.policy_optimizer.init(policy_params)
        critic_optimizer_state = self.critic_optimizer.init(critic_params)
        zs_encoder_optimizer_state = self.zs_encoder_optimizer.init(zs_encoder_params)
        za_encoder_optimizer_state = self.za_encoder_optimizer.init(za_encoder_params)
        zsa_encoder_optimizer_state = self.zsa_encoder_optimizer.init(zsa_encoder_params)
        dynamic_optimizer_state = self.dynamic_optimizer.init(dynamic_params)
        reward_optimizer_state = self.reward_optimizer.init(reward_params)
        termination_optimizer_state = self.termination_optimizer.init(termination_params)

        return WorldModelTrainingState(
            policy_params=policy_params,
            critic_params=critic_params,
            zs_encoder_params=zs_encoder_params,
            za_encoder_params=za_encoder_params,
            zsa_encoder_params=zsa_encoder_params,
            dynamic_params=dynamic_params,
            reward_params=reward_params,
            termination_params=termination_params,

            target_zs_encoder_params=target_zs_encoder_params,
            target_za_encoder_params=target_za_encoder_params,
            target_zsa_encoder_params=target_zsa_encoder_params,
            target_dynamic_params=target_dynamic_params,
            target_reward_params=target_reward_params,
            target_termination_params=target_termination_params,
            target_critic_params=target_critic_params,
            target_policy_params=target_policy_params,
            
            policy_optimizer_state=policy_optimzier_state,
            critic_optimizer_state=critic_optimizer_state,
            zs_encoder_optimizer_state=zs_encoder_optimizer_state,
            za_encoder_optimizer_state=za_encoder_optimizer_state,
            zsa_encoder_optimizer_state=zsa_encoder_optimizer_state,
            dynamic_optimizer_state=dynamic_optimizer_state,
            reward_optimizer_state=reward_optimizer_state,
            termination_optimizer_state=termination_optimizer_state,
            rp_state=RunningPercentileState(value=jnp.ones(1,)),
        ), random_key
        

    @partial(jax.jit, static_argnames=("self"))
    def zs_encode(
        self,
        zs_encoder_params: Params,
        obs: Observation,
    ) -> Latent: 
        z = self._zs_encoder.apply(zs_encoder_params, obs)
        return z
        
    @partial(jax.jit, static_argnames=("self",))
    def zsa_encode(
        self,
        za_encoder_params: Params,
        zsa_encoder_params: Params,
        zs: Latent,
        action: Action,
    ) -> Latent:
        za = self._za_encoder.apply(za_encoder_params, action)
        input_ = jnp.concatenate([zs, za], axis=-1)
        z = self._zsa_encoder.apply(zsa_encoder_params, input_)
        return z

        
    @partial(jax.jit, static_argnames=("self"))
    def next(
        self,
        dynamic_params: Params,
        zsa: Latent,
    ) -> Latent: 
        zs = self._dynamic.apply(dynamic_params, zsa)
        return zs


    @partial(jax.jit, static_argnames=("self"))
    def reward(
        self,
        reward_params: Params,
        zsa: Latent,
    ) -> Reward:
        r = self._reward.apply(reward_params, zsa)  
        return r # shape: (batch_size, num_bins)

    @partial(jax.jit, static_argnames=("self", "apply_sigmoid"))
    def termination(
        self,
        termination_params: Params, 
        zsa: Latent,
        apply_sigmoid: bool = False, 
    ) -> jnp.ndarray:
        ter = self._termination.apply(termination_params, zsa)
        if apply_sigmoid:
            ter = nn.sigmoid(ter)
        return ter

    @partial(jax.jit, static_argnames=("self"))
    def pi(
        self,
        policy_params: Params, 
        zs: Latent,
    ) -> Tuple[Action, jnp.ndarray]:
        pre_activ = self._policy.apply(policy_params, zs)
        actions = jax.nn.tanh(pre_activ)

        return actions, pre_activ
    

    @partial(jax.jit, static_argnames=("self"))
    def Q(
        self,
        critic_params: Params,
        zsa: Latent, 
    ) -> jnp.ndarray:

        qs = self._critic.apply(critic_params, zsa) 

        return qs # shape: (batch_size, num_bins, num_qs)
