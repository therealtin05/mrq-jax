from dataclasses import dataclass
from functools import partial
from typing import Callable, Any, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from dm_control import suite
import gymnasium as gym

from ml_collections import ConfigDict

from tqdm import tqdm

from MRQ.mrq import MRQ, MRQConfig, MRQTrainingState
from MRQ.buffers.transitions import Transition
from flashbax.buffers.trajectory_buffer import make_trajectory_buffer
from MRQ.common.mdp_utils import generate_unroll, get_mask_from_transitions
from MRQ.envs.dmcontrol import make_dmc_env

from MRQ.custom_types import RNGKey, TrainingState

@dataclass
class TrainerConfig:
    num_env_steps: int = 1_000_000
    num_warmstart_steps: int = 10000
    utd_ratio: float = 1.0
    replay_buffer_size: int = 1_000_000
    batch_size: int = 256
    eval_frequency: int = 10000
    num_eval_episodes: int = 100
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_frequency: int = 1024
    seed: int = 42
        
class Trainer:

    def __init__(self, trainer_config: TrainerConfig, 
                 mrq_config: MRQConfig,
                 env_config: ConfigDict):
        self._trainer_config = trainer_config
        self._mrq_config = mrq_config
        self._env_config = env_config

        ##############################
        # Environment setup
        ##############################
        def make_env(env_config, seed):
            def make_gym_env(env_id, seed):
                env = gym.make(env_id)
                env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.Autoreset(env)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            if env_config.backend == "gymnasium":
                return make_gym_env(env_config.env_name, seed)
            elif env_config.backend == "dmc":
                env = make_dmc_env(env_config.env_name, seed, env_config.dmc.obs_type)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.Autoreset(env)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            else:
                raise ValueError("Environment not supported:", env_config)

        if env_config.asynchronous:
            vector_env_cls = gym.vector.AsyncVectorEnv
        else:
            vector_env_cls = gym.vector.SyncVectorEnv
        env = vector_env_cls(
            [
                partial(make_env, env_config, seed)
                for seed in range(trainer_config.seed, trainer_config.seed + env_config.num_envs)
            ]
        )

        self._env = env
        self._observation_size = env.observation_space.shape[-1]
        self._action_size = env.action_space.shape[-1]
        self._eval_env = make_env(env_config, trainer_config.seed)
        # Define once, at __init__ or before the loop


        self.replay_buffer = make_trajectory_buffer(
            max_length_time_axis=trainer_config.replay_buffer_size // env_config.num_envs, # Maximum length of the buffer along the time axis. 
            min_length_time_axis=16, # Minimum length across the time axis before we can sample.
            sample_batch_size=mrq_config.batch_size, # Batch size of trajectories sampled from the buffer.
            add_batch_size=env_config.num_envs, # Batch size of trajectories added to the buffer.
            sample_sequence_length=mrq_config.enc_horizon, # Sequence length of trajectories sampled from the buffer. ### DEBUG
            period=1, # Period at which we sample trajectories from the buffer.
        )
        self.add_experience = jax.jit(self.replay_buffer.add)

        self.mrq = MRQ(
            config=mrq_config,
            observation_size=self._observation_size,
            action_size=self._action_size,
            replay_buffer=self.replay_buffer,
        )

        train_select_action_fn = partial(
            self.mrq.select_action,
            deterministic=False,
        )
        self.jitted_select_action_fn = jax.jit(train_select_action_fn)
        
        eval_select_action_fn = partial(
            self.mrq.select_action,
            deterministic=True,
        )
        self.jitted_eval_select_action_fn = jax.jit(eval_select_action_fn)


    def init(self) -> Tuple[MRQTrainingState, RNGKey]:
        random_key = jax.random.PRNGKey(self._trainer_config.seed)

        ### init buffer ###
        dummy_transition = Transition.init_dummy(
            self._observation_size, self._action_size
        )
        dummy_transition = jtu.tree_map(
            lambda x: x.squeeze(0), dummy_transition
        )
        buffer_state = self.replay_buffer.init(
            dummy_transition
        )
        mrq_state, random_key = self.mrq.init(
            buffer_state=buffer_state, random_key=random_key
        )

        return mrq_state, random_key


    # @partial(jax.jit, static_argnames=("self", "play_step_fn"))
    def evaluate(
        self, 
        training_state: TrainingState,
        random_key: RNGKey,
        play_step_fn: Callable,
    ) -> Tuple[Dict, RNGKey]:

        total_reward = []
        total_ep_len = []

        for _ in range(self._trainer_config.num_eval_episodes):
            obs, _ = self._eval_env.reset()
            eval_done = False
            while not eval_done:
                action, random_key = play_step_fn(
                        training_state,
                        obs,
                        random_key,
                    )
                
                eval_next_obs, reward, terminated, truncated, info = self._eval_env.step(np.array(action))
                obs = eval_next_obs  
                eval_done = terminated | truncated

            total_reward.append(info['episode']['r'])
            total_ep_len.append(info['episode']['l'])

        eval_metrics = {
            "eval/fitness_avg": np.mean(np.array(total_reward)),
            "eval/fitness_std": np.std(np.array(total_reward)),
            "eval/ep_length_avg": np.mean(np.array(total_ep_len)),
            "eval/ep_length_std":np.std(np.array(total_ep_len)),
        }

        return eval_metrics, random_key

                    


        
    @partial(jax.jit, static_argnames=("self", "num_updates"))
    def update(
        self,
        training_state,
        random_key,
        num_updates
    ) -> Tuple[TrainingState, RNGKey, Dict]:
        
        def scan_update(
                carry: Tuple[MRQTrainingState, RNGKey], unused: Any):
            training_state, random_key, metrics = self.mrq.update(*carry)
            return (training_state, random_key), metrics

        (training_state, random_key), train_metrics = jax.lax.scan(
            scan_update,
            (training_state, random_key),
            (),
            length=num_updates,
        )  
        return training_state, random_key, train_metrics



    def train(self, log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None):
        # init

        total_training_metrics = {}
        # Accumulate metric values across multiple update() calls until the next logging step.
        # We store a list per metric key so we can take a mean later.
        def _append_metrics(acc: Dict[str, list], new: Dict[str, Any]) -> Dict[str, list]:
            for k, v in new.items():
                acc.setdefault(k, []).append(jnp.asarray(v))
            return acc
        

        training_state, random_key = self.init()
        buffer_state = training_state.buffer_state  
        global_step = 0
        ep_count = 0
        times_logged = 0
        obs, _ = self._env.reset()

        # num_warmup_steps = int(
        #     1000 * self._env_config.num_envs * self._trainer_config.utd_ratio,
        # )
        num_warmup_steps = self._trainer_config.num_warmstart_steps
        pbar = tqdm(initial=0, total=self._trainer_config.num_env_steps)


        done = np.zeros(self._env_config.num_envs, dtype=bool)
        is_adding_exp = np.zeros(self._env_config.num_envs, dtype=bool)
        while global_step < self._trainer_config.num_env_steps:

            if global_step <= num_warmup_steps:
                action = self._env.action_space.sample()
            else:
                action, random_key = self.jitted_select_action_fn(
                    training_state,
                    obs,
                    random_key,
                )
                action = np.array(action)

            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = np.logical_or(terminated, truncated)
            transition = Transition(
                obs=jnp.array(obs),
                actions=jnp.array(action),
                rewards=jnp.array(reward).squeeze(),
                next_obs=jnp.array(next_obs),
                dones=jnp.array(done, dtype=jnp.float32).squeeze(),
                truncations=jnp.array(truncated, dtype=jnp.float32).squeeze(),
            )
            transition = jtu.tree_map(
                lambda x: jnp.expand_dims(x, 1), transition
            )   

            # due to gym autoreset bug, last transition is corupted unlike brax
            if not np.any(is_adding_exp):
                buffer_state = self.add_experience(buffer_state, transition)
                
            training_state = training_state.replace(buffer_state=buffer_state)

            obs = next_obs

            is_adding_exp = np.logical_or(terminated, truncated)

            global_step += self._env_config.num_envs

            if np.any(done):

                for ienv in range(self._env_config.num_envs):
                    if done[ienv]:
                        r = info['episode']['r'][ienv]
                        l = info['episode']['l'][ienv]
                        print(
                            f"Episode {ep_count}: r = {r:.2f}, l = {l}"
                        )
                        
                        if log_fn is not None:
                            log_fn(global_step + ienv, {
                                f"training/episode_return": r,
                                f"training/episode_length": l,
                            })

                        ep_count += 1


            if global_step >= num_warmup_steps:
                if global_step == num_warmup_steps:
                    # print('Pre-training on seed data...')
                    # num_updates = num_warmup_steps # follows TDMPC2
                    num_updates = max(1, int(self._env_config.num_envs * self._trainer_config.utd_ratio)) # exactly follows MRQ
                else:
                    num_updates = max(1, int(self._env_config.num_envs * self._trainer_config.utd_ratio))
                        
                training_state, random_key, train_metrics = self.update(
                    training_state, random_key, num_updates
                )

                # Average across the internal scan dimension (num_updates) so each update()
                # contributes one scalar per metric to the accumulator.
                train_metrics = jax.tree_util.tree_map(
                    lambda x: x.mean(), train_metrics
                )
                total_training_metrics = _append_metrics(total_training_metrics, train_metrics)

                if global_step // self._trainer_config.eval_frequency > times_logged:
                    eval_metrics, random_key = self.evaluate(
                        training_state, random_key, self.jitted_eval_select_action_fn
                    )

                    metrics = train_metrics | eval_metrics

                    # metrics = train_metrics

                    print(f"Env steps {global_step}")
                    for key, value in metrics.items():
                        print(f"\t{key}: {value}")

                    if log_fn is not None:
                        loggable_metrics = {
                            key: float(value)
                            for key, value in metrics.items()
                        }
                        loggable_metrics["env_steps"] = float(global_step)
                        log_fn(global_step, loggable_metrics)
                        
                    times_logged += 1
                    total_training_metrics = {}


            pbar.update(self._env_config.num_envs)

        pbar.close()


