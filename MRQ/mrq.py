from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Any, Dict

from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import linen as nn
import optax

from MRQ.world_model_mrq import WorldModel, WorldModelConfig, WorldModelTrainingState
from MRQ.custom_types import Params, RNGKey, Observation, Latent, Action, TrainingState, Metrics
from MRQ.common.mdp_utils import multi_step_reward
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, make_trajectory_buffer, TrajectoryBufferState, sample as sample_trajectory_buffer

from MRQ.common.math_utils import two_hot, two_hot_inv, soft_ce

def sg(x): return jax.tree.map(jax.lax.stop_gradient, x)

# Normalize input values using a running scale of the range between a given range of percentiles.
def percentile_normalization(x: jnp.ndarray,
                             prev_scale: jnp.ndarray,
                             percentile_range: jnp.ndarray = jnp.array([5, 95]),
                             tau: float = 0.01) -> jnp.ndarray:
  # Compute percentiles for the input values.
  percentiles = jnp.percentile(x, percentile_range)
  scale = percentiles[1] - percentiles[0]

  return tau * scale + (1 - tau) * prev_scale


@dataclass
class MRQConfig:
    # ReplayBuffer

    batch_size: int = 256
    num_qs: int = 2
    num_bins: int = 1
    low: int = -10
    high: int = 10
    num_enc_layer: int = 2
    simnorm_dim: int = 8
    enc_horizon: int = 5
    rl_horizon: int = 3

    # MPC
    mpc: bool = True
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    horizon: int = 3
    min_std: float = 0.05
    max_std: int = 2
    temperature: float = 0.5
    discount: float = 0.99 
    episodic: bool = True

    # TD3
    exploration_noise: float = 0.2
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3
    pre_activation_weight: float = 1e-5

    # Hyperparam
    reward_coef: float = 0.1
    termination_coef: float = 0.1
    consistency_coef: float = 20.0
    bc_coef: float = 0.0
    rho: float = 0.5
    lr: float = 3e-4
    enc_lr: float = 1e-4

    # Architecture
    zs_encoder_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
    za_encoder_hidden_layer_sizes: Tuple[int, ...] = (256,)
    zsa_encoder_hidden_layer_sizes: Tuple[int, ...] = (512, 512)
    critic_hidden_layer_sizes: Tuple[int, ...] = (512, 512, 512)
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 512)

    latent_dim: int = 512
    max_grad_norm: float = 100.0 # set to 0.0 means not use gradient clipping

    target_update_freq: int = 250


class MRQTrainingState(TrainingState):
    wm_state: WorldModelTrainingState
    reward_scale: jnp.ndarray
    target_reward_scale: jnp.ndarray
    buffer_state: TrajectoryBufferState
    steps: jnp.ndarray


class MRQ(nn.Module):
    def __init__(self, config: MRQConfig, observation_size: int, action_size: int, replay_buffer: TrajectoryBuffer):
        self._config = config
        self._observation_size = observation_size
        self._action_size = action_size
        self._replay_buffer = replay_buffer

        self.sample_buffer_enc = partial(sample_trajectory_buffer, batch_size=config.batch_size, sequence_length=config.enc_horizon, period=1)
        self.sample_buffer_rl = partial(sample_trajectory_buffer, batch_size=config.batch_size, sequence_length=config.rl_horizon, period=1)

        wm_config = WorldModelConfig(
            num_qs=config.num_qs,
            num_bins=config.num_bins,
            low=config.low,
            high=config.high,
            simnorm_dim=config.simnorm_dim,
            lr=config.lr,
            enc_lr=config.enc_lr,
            zs_encoder_hidden_layer_sizes=config.zs_encoder_hidden_layer_sizes,
            za_encoder_hidden_layer_sizes=config.za_encoder_hidden_layer_sizes,
            zsa_encoder_hidden_layer_sizes=config.zsa_encoder_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.critic_hidden_layer_sizes,
            policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
            latent_dim=config.latent_dim,
            max_grad_norm=config.max_grad_norm,
        )
        self._wm = WorldModel(config=wm_config, action_size=action_size, 
                                      observation_size=observation_size)
    


    def init(self, buffer_state: TrajectoryBufferState, random_key: RNGKey) -> Tuple[MRQTrainingState, RNGKey]:
        wm_state, random_key = self._wm.init(random_key)
        return MRQTrainingState(
            wm_state=wm_state,
            buffer_state=buffer_state,
            steps=jnp.zeros(()),
            reward_scale=jnp.array(1.0),
            target_reward_scale=jnp.array(0.0)
        ), random_key
    


    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def select_action(
            self,
            training_state: MRQTrainingState,
            obs: Observation,
            random_key: RNGKey,
            deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, RNGKey]:
        wm_state = training_state.wm_state
        zs = self._wm.zs_encode(wm_state.zs_encoder_params, obs)
        action, _ = self._wm.pi(
            wm_state.policy_params,
            zs,
        )
        if not deterministic:
            random_key, subkey = jax.random.split(random_key)
            action += jax.random.normal(subkey, action.shape) * self._config.exploration_noise
            action = jnp.clip(action, -1, 1)
        return action, random_key



    @partial(jax.jit, static_argnames=("self"))
    def update_encoder(
            self,
            training_state: MRQTrainingState,
            random_key: RNGKey,
        ) -> Tuple[MRQTrainingState, RNGKey, Dict[str, Any]]:

        random_key, random_key = jax.random.split(random_key)
        world_model_key, policy_key, buffer_key = jax.random.split(random_key, 3)

        wm_state = training_state.wm_state
        batch = self.sample_buffer_enc(
            training_state.buffer_state, buffer_key
        )
        experience = jtu.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), batch.experience
        ) # (horizon, batch_size, ...)
        observations = experience.obs
        actions = experience.actions
        rewards = experience.rewards
        next_observations = experience.next_obs
        terminated = jnp.logical_and(experience.dones, (jnp.logical_not(experience.truncations)))
        truncated = experience.truncations

        def world_model_loss_fn(
                zs_encoder_params: Params,
                za_encoder_params: Params,
                zsa_encoder_params: Params,
                dynamics_params: Params,
                reward_params: Params,
                termination_params: Params,
            ) -> Tuple[jnp.ndarray, Dict[str, Any]]:

            ### borrow from tdmpc2
            # lam = self._config.rho**jnp.arange(self._config.enc_horizon)
            # lam /= jnp.sum(lam)

            ### following mrq, no deviding by enc_horizon
            lam = jnp.ones(self._config.enc_horizon) ### 

            ###########################################################
            # Encoder forward pass
            ###########################################################
            all_obs = jax.tree.map(
                lambda x, y: jnp.stack([x, y], axis=0),
                observations, next_observations
            ) 
            all_zs = self._wm.zs_encode(
                zs_encoder_params, obs=all_obs,
            )
            encoder_zs = jax.tree.map(lambda x: x[0], all_zs)
            encoder_next_zs = jax.tree.map(lambda x: x[1], all_zs)

            # encoder_zs = self._wm.zs_encode(zs_encoder_params, observations)
            # next_zs = self._wm.zs_encode(zs_encoder_params, next_observations)
            target_next_zs = self._wm.zs_encode(wm_state.target_zs_encoder_params, next_observations)

            ###########################################################
            # Latent rollout (dynamics + consistency loss)
            ###########################################################
            done = jnp.logical_or(terminated, truncated) # (horizon, batch_size)
            latent_zs = jnp.zeros(
                (self._config.enc_horizon+1, self._config.batch_size, self._config.latent_dim)
            )
            latent_zs = latent_zs.at[0].set(encoder_zs[0])

            finished = jnp.zeros((self._config.enc_horizon+1, self._config.batch_size))
            finished = finished.at[:-1].set(done)
            finished = jnp.clip(jnp.cumsum(finished, axis=0), 0, 1).astype(bool)
            
            dynamic_mask = jnp.logical_not(finished)
            r_c_p_t_mask = jnp.logical_not(jnp.roll(finished, 1, axis=0).at[0].set(0))


            def scan_imagine_z(
                    carry: Latent, xs: Action
                ) -> Tuple[Latent, Tuple[Latent, Latent]]:
                zsa = self._wm.zsa_encode(
                    za_encoder_params,
                    zsa_encoder_params,
                    carry,
                    xs,
                )
                next_zs = self._wm.next(dynamics_params, zsa)

                return next_zs, (zsa, next_zs)
            
            _, (imagined_zsa, imagined_next_zs) = jax.lax.scan(
                scan_imagine_z,
                encoder_zs[0],
                actions,
            ) # next_zs: (horizon, batch_size, latent_size) timestep: 2 -> horizon + 1, zsa: (horizon, batch_size, latent_size)  timestep: 1 -> horizon
            # target_next_zs: (horizon, batch_size, latent_size) timestep: 2 -> horizon + 1, target_zsa: (horizon, batch_size, latent_size)  timestep: 1 -> horizon

            latent_zs = latent_zs.at[1:].set(imagined_next_zs) 

            consistency_loss = jnp.sum(
                lam * jnp.mean(
                    (imagined_next_zs - sg(target_next_zs))**2, 
                    # (imagined_next_zs - sg(encoder_next_zs))**2, 
                    where=dynamic_mask[:-1][..., None], 
                    axis=(-1, -2)
                )
            ) # lam: (horizon,), mean result: (horizon, batch_size)


            ###########################################################
            # Reward loss
            ###########################################################
            reward_logits = self._wm.reward(
                reward_params, imagined_zsa,
            ) # (horizon, batch_size, num_bins) 
            reward_loss = jnp.sum(
                lam[:, None] * soft_ce(
                    pred=reward_logits,
                    target=two_hot(rewards, self._config.low, self._config.high, self._config.num_bins),
                ), axis=0, where=r_c_p_t_mask[:-1]
            ).mean()

            ###########################################################
            # Termination loss
            ###########################################################
            if self._config.episodic:
                termination_logits = self._wm.termination(termination_params, imagined_zsa).squeeze(-1)
                termination_loss = jnp.sum(
                    lam[:, None] * optax.sigmoid_binary_cross_entropy(termination_logits, terminated),
                    axis = 0,
                    where=r_c_p_t_mask[:-1]
                ).mean()

                pred_termination_binary = (jax.nn.sigmoid(termination_logits) > 0.5).astype(jnp.float32)
                termination_float = terminated.astype(jnp.float32)

                tp = jnp.sum(pred_termination_binary * termination_float * r_c_p_t_mask[:-1])
                fp = jnp.sum(pred_termination_binary * (1 - termination_float) * r_c_p_t_mask[:-1])
                fn = jnp.sum((1 - pred_termination_binary) * termination_float * r_c_p_t_mask[:-1])

                precision = tp / (tp + fp + 1e-8)
                recall    = tp / (tp + fn + 1e-8)
                f1_score  = 2 * precision * recall / (precision + recall + 1e-8)

            else:
                pred_termination_binary = jnp.zeros((1,))
                termination_float = jnp.zeros((1,))
                precision = 0.0
                recall    = 0.0
                f1_score  = 0.0
                termination_loss = 0.0

            total_loss = (
                self._config.consistency_coef * consistency_loss +
                self._config.reward_coef * reward_loss +
                self._config.termination_coef * termination_loss
            )

            return total_loss, {
                'losses/consistency': consistency_loss,
                'losses/reward': reward_loss,
                'losses/termination': termination_loss,
                'losses/total_loss': total_loss,
                "metrics/termination_precision": precision,
                "metrics/termination_recall": recall,
                "metrics/termination_f1": f1_score,
                "metrics/termination_positive_rate": jnp.mean(termination_float),   # monitors class imbalance
                "metrics/termination_pred_positive_rate": jnp.mean(pred_termination_binary),
            }

        # Update world model
        (zs_encoder_grads, za_encoder_grads, zsa_encoder_grads, dynamics_grads, reward_grads, termination_grads), model_info = jax.grad(
            world_model_loss_fn, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)(
                wm_state.zs_encoder_params,
                wm_state.za_encoder_params,
                wm_state.zsa_encoder_params,
                wm_state.dynamic_params,
                wm_state.reward_params,
                wm_state.termination_params if self._config.episodic else None
        )


        zs_encoder_updates, zs_encoder_optimizer_state = self._wm.zs_encoder_optimizer.update(zs_encoder_grads, wm_state.zs_encoder_optimizer_state, wm_state.zs_encoder_params)
        zsa_encoder_updates, zsa_encoder_optimizer_state = self._wm.zsa_encoder_optimizer.update(zsa_encoder_grads, wm_state.zsa_encoder_optimizer_state, wm_state.zsa_encoder_params)
        za_encoder_updates, za_encoder_optimizer_state = self._wm.za_encoder_optimizer.update(za_encoder_grads, wm_state.za_encoder_optimizer_state, wm_state.za_encoder_params)

        new_zs_encoder_params = optax.apply_updates(wm_state.zs_encoder_params, zs_encoder_updates)
        new_zsa_encoder_params = optax.apply_updates(wm_state.zsa_encoder_params, zsa_encoder_updates)
        new_za_encoder_params = optax.apply_updates(wm_state.za_encoder_params, za_encoder_updates)

        dynamic_updates, dynamic_optimizer_state = self._wm.dynamic_optimizer.update(dynamics_grads, wm_state.dynamic_optimizer_state, wm_state.dynamic_params)
        new_dynamic_params = optax.apply_updates(wm_state.dynamic_params, dynamic_updates)

        reward_updates, reward_optimizer_state = self._wm.reward_optimizer.update(reward_grads, wm_state.reward_optimizer_state, wm_state.reward_params)
        new_reward_params = optax.apply_updates(wm_state.reward_params, reward_updates)


        if self._config.episodic:
            termination_updates, termination_optimizer_state = self._wm.termination_optimizer.update(termination_grads, wm_state.termination_optimizer_state, wm_state.termination_params)
            new_termination_params = optax.apply_updates(wm_state.termination_params, termination_updates)
        else:
            termination_optimizer_state = wm_state.termination_optimizer_state
            new_termination_params = wm_state.termination_params



        wm_state = wm_state.replace(
            zs_encoder_params=new_zs_encoder_params,
            za_encoder_params=new_za_encoder_params,
            zsa_encoder_params=new_zsa_encoder_params,
            reward_params=new_reward_params,
            dynamic_params=new_dynamic_params,
            termination_params=new_termination_params,

            zs_encoder_optimizer_state=zs_encoder_optimizer_state,
            za_encoder_optimizer_state=za_encoder_optimizer_state,
            zsa_encoder_optimizer_state=zsa_encoder_optimizer_state, 
            reward_optimizer_state=reward_optimizer_state,
            dynamic_optimizer_state=dynamic_optimizer_state,
            termination_optimizer_state=termination_optimizer_state,
        )
        
        training_state = training_state.replace(
            wm_state=wm_state
        )

        # Compute gradient norms for logging
        grad_norms = {
            'grad_norm/zs_encoder': optax.global_norm(zs_encoder_grads),
            'grad_norm/za_encoder': optax.global_norm(za_encoder_grads),
            'grad_norm/zsa_encoder': optax.global_norm(zsa_encoder_grads),
            'grad_norm/dynamic': optax.global_norm(dynamics_grads),
            'grad_norm/reward': optax.global_norm(reward_grads),
            'grad_norm/termination': optax.global_norm(termination_grads) if self._config.episodic else 0.0,
        }


        return training_state, random_key, model_info | grad_norms


    def update_rl(
        self,
        training_state: MRQTrainingState,
        random_key: RNGKey,
    ) -> Tuple[MRQTrainingState, RNGKey, Metrics]:
        
        wm_state = training_state.wm_state
        random_key, subkey = jax.random.split(random_key)
        critic_key, ensemble_key, policy_key, buffer_key = jax.random.split(subkey, 4)

        wm_state = training_state.wm_state
        batch = self.sample_buffer_rl(
            training_state.buffer_state, buffer_key
        )
        experience = jtu.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), batch.experience
        )
        observations = experience.obs[0]  # (batch, obs_dim)
        actions = experience.actions[0]  # (batch, act_dim)
        rewards = experience.rewards
        next_observations = experience.next_obs[-1] # takes last next observations only
        terminated = jnp.logical_and(experience.dones, (jnp.logical_not(experience.truncations)))
        truncated = experience.truncations 
    
        mask = jnp.cumsum(jnp.logical_or(terminated, truncated), axis=0)
        mask = jnp.roll(mask, 1, axis=0)
        continues = jnp.clip(mask.at[0].set(0), 0, 1) # (horizon, batch_size)
        continues = jnp.logical_not(continues) # (horizon, batch_size) 
        cum_reward, discounts = multi_step_reward(rewards, continues, self._config.discount) # (batch_size) , (1, )

        encoded_zs = self._wm.zs_encode(wm_state.zs_encoder_params, observations) # (batch_size, latent_dim)

        ##########################
        # critic update
        ##########################
        def _critic_loss(critic_params):
            encoded_zsa = self._wm.zsa_encode(
                wm_state.za_encoder_params, wm_state.zsa_encoder_params, encoded_zs, actions
            )  # (batch, latent_dim)
            target_encoded_next_zs = self._wm.zs_encode(wm_state.target_zs_encoder_params, next_observations) # (batch_size, latent_dim)
            
            next_action, _ = self._wm.pi(wm_state.target_policy_params, target_encoded_next_zs) # (batch_size, action_dim)
            next_action += jnp.clip(jax.random.normal(critic_key, next_action.shape) * self._config.target_policy_noise, -self._config.noise_clip, self._config.noise_clip)
            next_action = jnp.clip(next_action, -1, 1)
            target_encoded_next_zsa = self._wm.zsa_encode(
                wm_state.target_za_encoder_params,
                wm_state.target_zsa_encoder_params,
                target_encoded_next_zs,
                next_action,
            )  # (batch, latent_dim)
            
            Qs = self._wm.Q(
                wm_state.target_critic_params,
                target_encoded_next_zsa,
            ).squeeze() # (batch_size, num_qs)
            
            Q = Qs[..., :2].min(axis=-1)  # (batch_size)
            td_targets = (cum_reward + (1 - terminated[-1]) * discounts * Q * training_state.target_reward_scale)/training_state.reward_scale  # (batch,)
            Q_logits = self._wm.Q(
                critic_params, encoded_zsa,
            ).squeeze() # (batch_size, num_qs)
            value_loss = jnp.mean(optax.losses.huber_loss(Q_logits, sg(td_targets[..., None])))
            return value_loss, {"losses/value_loss": value_loss}
        
        critic_grads, critic_info = jax.grad(_critic_loss, has_aux=True)(
            wm_state.critic_params
        )

        critic_updates, critic_optimizer_state = self._wm.critic_optimizer.update(critic_grads, wm_state.critic_optimizer_state, wm_state.critic_params)
        new_critic_params = optax.apply_updates(wm_state.critic_params, critic_updates)


        ###########################
        # policy update
        ###########################
        def policy_loss_fn(actor_params: Params):
            sampled_actions, pre_activ = self._wm.pi(actor_params, encoded_zs)  

            zsa = self._wm.zsa_encode(
                wm_state.za_encoder_params, wm_state.zsa_encoder_params, encoded_zs, sampled_actions
            )  # (batch, latent_dim)
            Qs = self._wm.Q(
                new_critic_params, zsa,
            ).squeeze()
            Q = Qs.mean(axis=-1) # (batch_size,)
            rl_loss = -jnp.mean(Q)
            # bc loss
            if self._config.bc_coef != 0.0:   
                bc_loss = self._config.bc_coef * jnp.mean(
                    (sampled_actions - actions) ** 2,
                )
            else:
                bc_loss = 0.0
                
            policy_loss = rl_loss + bc_loss + self._config.pre_activation_weight * jnp.mean(jnp.square(pre_activ))

            return policy_loss, {            
                'losses/bc': bc_loss,
                'losses/rl': rl_loss,
                'losses/policy': policy_loss,
            }

        policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
            wm_state.policy_params
        )
        policy_updates, policy_optimizer_state = self._wm.policy_optimizer.update(policy_grads, wm_state.policy_optimizer_state, wm_state.policy_params)
        new_policy_params = optax.apply_updates(wm_state.policy_params, policy_updates)

        grad_norms = {                
            'grad_norm/policy': optax.global_norm(policy_grads),
            'grad_norm/critic': optax.global_norm(critic_grads),
            "metrics/reward_scale": training_state.reward_scale,
        }

        wm_state = wm_state.replace(
            critic_params=new_critic_params,
            policy_params=new_policy_params,
            critic_optimizer_state=critic_optimizer_state,
            policy_optimizer_state=policy_optimizer_state
        )

        training_state = training_state.replace(
            wm_state=wm_state,
            steps=training_state.steps + 1
        )

        return training_state, random_key, critic_info | policy_info | grad_norms


    @partial(jax.jit, static_argnames=("self"))
    def update(self,
                training_state: MRQTrainingState,
                random_key: RNGKey,
                ) -> Tuple[MRQTrainingState, RNGKey, Dict[str, Any]]:

        wm_state = training_state.wm_state
        encoder_metrics = {
            'losses/consistency': 0.0,
            'losses/reward': 0.0,
            'losses/termination': 0.0,
            'losses/total_loss': 0.0,
            "metrics/termination_precision": 0.0,
            "metrics/termination_recall": 0.0,
            "metrics/termination_f1": 0.0,
            "metrics/termination_positive_rate": 0.0,   # monitors class imbalance
            "metrics/termination_pred_positive_rate": 0.0,
            'grad_norm/zs_encoder': 0.0,
            'grad_norm/za_encoder': 0.0,
            'grad_norm/zsa_encoder': 0.0,
            'grad_norm/dynamic': 0.0,
            'grad_norm/reward': 0.0,
            'grad_norm/termination': 0.0,
        }

        def update_both(operand):
            training_state, random_key = operand
            wm_state = training_state.wm_state
            wm_state = wm_state.replace(
                target_policy_params =jtu.tree_map(lambda x: x, wm_state.policy_params),
                target_critic_params = jtu.tree_map(lambda x: x, wm_state.critic_params),
                target_zs_encoder_params = jtu.tree_map(lambda x: x, wm_state.zs_encoder_params),
                target_zsa_encoder_params = jtu.tree_map(lambda x: x, wm_state.zsa_encoder_params),
                target_za_encoder_params = jtu.tree_map(lambda x: x, wm_state.za_encoder_params),
                target_dynamic_params = jtu.tree_map(lambda x: x, wm_state.dynamic_params),
                target_reward_params = jtu.tree_map(lambda x: x, wm_state.reward_params),
            )
            rewards = training_state.buffer_state.experience.rewards
            current_index = training_state.buffer_state.current_index
            mask = jnp.arange(rewards.shape[1]) < current_index
            new_reward_scale = jnp.mean(jnp.abs(rewards), where=mask)

            training_state = training_state.replace(
                wm_state=wm_state,
                reward_scale=new_reward_scale,
                target_reward_scale=training_state.reward_scale,
            )
            # encoder_updates 
            def _scan_update_encoder(
                carry, zs
            ):
                training_state, random_key = carry
                training_state, random_key, metrics = self.update_encoder(training_state, random_key)
                return (training_state, random_key), metrics

            (training_state, random_key), encoder_metrics = jax.lax.scan(
                _scan_update_encoder,
                (training_state, random_key),
                (),
                self._config.target_update_freq
            )
            encoder_metrics = jtu.tree_map(
                lambda x: jnp.mean(x), encoder_metrics
            )
            training_state, random_key, rl_metrics = self.update_rl(
                training_state, random_key
            )
            return training_state, random_key, rl_metrics | encoder_metrics 

        def update_rl_only(operand):
            training_state, random_key = operand
            training_state, random_key, rl_metrics = self.update_rl(
                training_state, random_key
            )
            return training_state, random_key, rl_metrics | encoder_metrics

        training_state, random_key, metrics = jax.lax.cond(
            training_state.steps % self._config.target_update_freq == 0,
            update_both,
            update_rl_only,
            operand=(training_state, random_key)
        )

        return training_state, random_key, metrics
