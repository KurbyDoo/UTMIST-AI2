from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO
from pygame.locals import QUIT
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from environment.environment import ActHelper, AirTurnaroundState, Animation, AnimationSprite2D, AttackState, BackDashState, Camera, CameraResolution, Capsule, CapsuleCollider, Cast, CastFrameChangeHolder, CasterPositionChange, CasterVelocityDampXY, CasterVelocitySet, CasterVelocitySetXY, CompactMoveState, DashState, DealtPositionTarget, DodgeState, Facing, GameObject, Ground, GroundState, HurtboxPositionChange, InAirState, KOState, KeyIconPanel, KeyStatus, MalachiteEnv, MatchStats, MoveManager, MoveType, ObsHelper, Particle, Player, PlayerInputHandler, PlayerObjectState, PlayerStats, Power, RenderMode, Result, Signal, SprintingState, Stage, StandingState, StunState, Target, TauntState, TurnaroundState, UIHandler, WalkingState, WarehouseBrawl, hex_to_rgb

import warnings
from typing import TYPE_CHECKING, Any, Generic, \
    SupportsFloat, TypeVar, Type, Optional, List, Dict, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, MISSING
from collections import defaultdict
from functools import partial
from typing import Tuple, Any


from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

import gdown
import os
import math
import random
import shutil
import json
import logging
import time

import numpy as np
import torch
from torch import nn

import gymnasium
from gymnasium import spaces

import pygame
import pygame.gfxdraw
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d

import cv2
import skimage.transform as st
import skvideo
import skvideo.io
from IPython.display import Video

from stable_baselines3.common.monitor import Monitor


# Logging information
def setup_logger(log_dir: str, name: str) -> logging.Logger:
    """Set up a logger that writes to a file in the log directory."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

# ## Agents

# ### Agent Abstract Base Class

# In[ ]:


SelfAgent = TypeVar("SelfAgent", bound="Agent")


class Agent(ABC):

    def __init__(
        self,
        file_path: Optional[str] = None
    ):

        # If no supplied file_path, load from gdown (optional file_path returned)
        if file_path is None:
            file_path = self._gdown()

        self.file_path: Optional[str] = file_path
        self.initialized = False

    def get_env_info(self, env):
        if isinstance(env, Monitor):
            self_env = env.env
        else:
            self_env = env
        self.observation_space = self_env.observation_space
        self.obs_helper = self_env.obs_helper
        self.action_space = self_env.action_space
        self.act_helper = self_env.act_helper
        self.env = env
        self._initialize()
        self.initialized = True

    def get_num_timesteps(self) -> int:
        if hasattr(self, 'model'):
            return self.model.num_timesteps
        else:
            return 0

    def update_num_timesteps(self, num_timesteps: int) -> None:
        if hasattr(self, 'model'):
            self.model.num_timesteps = num_timesteps

    @abstractmethod
    def predict(self, obs) -> spaces.Space:
        pass

    def save(self, file_path: str) -> None:
        return

    def reset(self) -> None:
        return

    def _initialize(self) -> None:
        """

        """
        return

    def _gdown(self) -> Optional[str]:
        """
        Loads the necessary file from Google Drive, returning a file path.
        Or, returns None, if the agent does not require loaded files.

        :return:
        """
        return


# ### Agent Classes

# In[ ]:


class ConstantAgent(Agent):
    '''
    ConstantAgent:
    - The ConstantAgent simply is in an IdleState (action_space all equal to zero.)
    As such it will not do anything, DON'T use this agent for your training.
    '''

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action


class RandomAgent(Agent):
    '''
    RandomAgent:
    - The RandomAgent (as it name says) simply samples random actions.
    NOT used for training
    '''

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


# ## StableBaselines3 Integration

# ### Reward Configuration

# In[ ]:

@dataclass
class RewTerm():
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """


# In[ ]:


class RewardManager():
    """Reward terms for the MDP."""

    # (1) Constant running reward
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]] = None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]] = None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0

    def subscribe_signals(self, env) -> None:
        if self.signal_subscriptions is None:
            return
        for _, (name, term_cfg) in self.signal_subscriptions.items():
            getattr(env, name).connect(partial(self._signal_func, term_cfg))

    def _signal_func(self, term_cfg: RewTerm, *args, **kwargs):
        term_partial = partial(term_cfg.func, **term_cfg.params)
        self.collected_signal_rewards += term_partial(
            *args, **kwargs) * term_cfg.weight

    def process(self, env, dt) -> float:
        # reset computation
        reward_buffer = 0.0
        # iterate over all the reward terms
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight
                # update total reward
                reward_buffer += value

        reward = reward_buffer + self.collected_signal_rewards
        self.collected_signal_rewards = 0.0

        self.total_reward += reward

        log = env.logger[0]
        log['reward'] = f'{reward_buffer:.3f}'
        log['total_reward'] = f'{self.total_reward:.3f}'
        env.logger[0] = log
        return reward

    def reset(self):
        self.total_reward = 0
        self.collected_signal_rewards


# ### Save, Self-play, and Opponents

# In[ ]:


class SelfPlayHandler(ABC):
    """Handles self-play."""

    def __init__(self, agent_partial: partial, save_path: str, run_name: str):
        self.agent_partial = agent_partial
        self.save_path = save_path
        self.run_name = run_name
        self.experiment_path = os.path.join(self.save_path, self.run_name)
        self.env = None  # This will be set by OpponentsCfg

    def get_model_from_path(self, path) -> Agent:
        if path:
            try:
                opponent = self.agent_partial(file_path=path)
            except FileNotFoundError:
                print(
                    f"Warning: Self-play file {path} not found. Defaulting to constant agent.")
                opponent = ConstantAgent()
        else:
            print("Warning: No self-play model saved. Defaulting to constant agent.")
            opponent = ConstantAgent()

        if self.env:
            opponent.get_env_info(self.env)
        else:
            print(
                "Warning: SelfPlayHandler does not have an environment reference. Cannot initialize opponent.")

        return opponent

    def get_all_models(self) -> List[str]:
        if not os.path.exists(self.experiment_path):
            return []
        history = [os.path.join(self.experiment_path, f) for f in os.listdir(
            self.experiment_path) if os.path.isfile(os.path.join(self.experiment_path, f))]
        history = [f for f in history if f.endswith('.zip')]
        if len(history) > 0:
            history.sort(key=lambda x: int(
                os.path.basename(x).split('_')[-2].split('.')[0]))
        return history

    @abstractmethod
    def get_opponent(self) -> Agent:
        pass


class SelfPlayLatest(SelfPlayHandler):
    def __init__(self, agent_partial: partial, save_path: str, run_name: str):
        super().__init__(agent_partial, save_path, run_name)

    def get_opponent(self) -> Agent:
        all_models = self.get_all_models()
        chosen_path = all_models[-1] if all_models else None
        return self.get_model_from_path(chosen_path)


class SelfPlayRandom(SelfPlayHandler):
    def __init__(self, agent_partial: partial, save_path: str, run_name: str):
        super().__init__(agent_partial, save_path, run_name)

    def get_opponent(self) -> Agent:
        all_models = self.get_all_models()
        chosen_path = random.choice(all_models) if all_models else None
        return self.get_model_from_path(chosen_path)


@dataclass
class OpponentsCfg():
    """Configuration for opponents.

    Args:
        swap_steps (int): Number of steps between swapping opponents.
        opponents (dict): Dictionary specifying available opponents and their selection probabilities.
    """
    swap_steps: int = 10_000
    opponents: dict[str, Any] = field(default_factory=lambda: {
        'random_agent': (0.8, partial(RandomAgent)),
        'constant_agent': (0.2, partial(ConstantAgent)),
        # 'recurrent_agent': (0.1, partial(RecurrentPPOAgent, file_path='skibidi')),
    })

    def validate_probabilities(self) -> None:
        total_prob = sum(prob if isinstance(prob, float)
                         else prob[0] for prob in self.opponents.values())

        if abs(total_prob - 1.0) > 1e-5:
            print(
                f"Warning: Probabilities do not sum to 1 (current sum = {total_prob}). Normalizing...")
            self.opponents = {
                key: (value / total_prob if isinstance(value, float)
                      else (value[0] / total_prob, value[1]))
                for key, value in self.opponents.items()
            }

    def process(self) -> None:
        pass

    def on_env_reset(self) -> Agent:

        agent_name = random.choices(
            list(self.opponents.keys()),
            weights=[prob if isinstance(prob, float) else prob[0]
                     for prob in self.opponents.values()]
        )[0]

        # If self-play is selected, return the trained model
        log_prefix = f"[Agent {self.env.rank}]" if self.env.rank is not None else "[AGENT]"
        print(f'{log_prefix} Selected {agent_name}')
        if agent_name == "self_play":
            selfplay_handler: SelfPlayHandler = self.opponents[agent_name][1]
            selfplay_handler.env = self.env  # Ensure the handler has the env reference
            return selfplay_handler.get_opponent()
        else:
            # Otherwise, return an instance of the selected agent class
            opponent = self.opponents[agent_name][1]()

        opponent.get_env_info(self.env)
        return opponent


# ### Self-Play Warehouse Brawl

# In[ ]:


class SelfPlayWarehouseBrawl(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 reward_manager: Optional[RewardManager] = None,
                 opponent_cfg: OpponentsCfg = OpponentsCfg(),
                 resolution: CameraResolution = CameraResolution.LOW,
                 rank: Optional[int] = None):
        """
        Initializes the environment.

        Args:
            reward_manager (Optional[RewardManager]): Reward manager.
            opponent_cfg (OpponentCfg): Configuration for opponents.
            resolution (CameraResolution): Rendering resolution.
            rank (Optional[int]): The process rank for logging in parallel environments.
        """
        super().__init__()

        self.reward_manager = reward_manager
        self.opponent_cfg = opponent_cfg
        self.resolution = resolution
        self.rank = rank

        self.games_done = 0

        # Give OpponentCfg references, and normalize probabilities
        self.opponent_cfg.env = self
        self.opponent_cfg.validate_probabilities()

        self.raw_env = WarehouseBrawl(resolution=resolution, train_mode=True)
        self.action_space = self.raw_env.action_space
        self.act_helper = self.raw_env.act_helper
        self.observation_space = self.raw_env.observation_space
        self.obs_helper = self.raw_env.obs_helper

        self.opponent_agent = None
        self.opponent_obs = None

    def step(self, action):

        full_action = {
            0: action,
            1: self.opponent_agent.predict(self.opponent_obs),
        }

        observations, rewards, terminated, truncated, info = self.raw_env.step(
            full_action)

        self.opponent_obs = observations[1]

        if self.reward_manager is None:
            reward = rewards[0]
        else:
            reward = self.reward_manager.process(self.raw_env, 1 / 30.0)

        return observations[0], reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset MalachiteEnv
        observations, info = self.raw_env.reset()

        if self.reward_manager:
            self.reward_manager.reset()

        # Select agent
        new_agent: Agent = self.opponent_cfg.on_env_reset()
        if new_agent is not None:
            self.opponent_agent: Agent = new_agent
        self.opponent_obs = observations[1]

        self.games_done += 1

        return observations[0], info

    def render(self):
        img = self.raw_env.render()
        return img

    def close(self):
        self.raw_env.close()


# ## Run Match

# In[ ]:


def run_match(agent_1: Agent | partial,
              agent_2: Agent | partial,
              max_timesteps=30*90,
              video_path: Optional[str] = None,
              agent_1_name: Optional[str] = None,
              agent_2_name: Optional[str] = None,
              resolution=CameraResolution.LOW,
              reward_manager: Optional[RewardManager] = None,
              train_mode=False
              ) -> MatchStats:
    # Initialize env

    env = WarehouseBrawl(resolution=resolution, train_mode=train_mode)
    observations, infos = env.reset()
    obs_1 = observations[0]
    obs_2 = observations[1]
    print("RUN MATCH IS RUNNING")
    if reward_manager is not None:
        reward_manager.reset()
        reward_manager.subscribe_signals(env)

    if agent_1_name is None:
        agent_1_name = 'agent_1'
    if agent_2_name is None:
        agent_2_name = 'agent_2'

    env.agent_1_name = agent_1_name
    env.agent_2_name = agent_2_name

    writer = None
    if video_path is None:
        print("video_path=None -> Not rendering")
    else:
        print(f"video_path={video_path} -> Rendering")
        # Initialize video writer
        writer = skvideo.io.FFmpegWriter(video_path, outputdict={
            '-vcodec': 'libx264',  # Use H.264 for Windows Media Player
            '-pix_fmt': 'yuv420p',  # Compatible with both WMP & Colab
            '-preset': 'fast',  # Faster encoding
            '-crf': '20',  # Quality-based encoding (lower = better quality)
            '-r': '30',  # Frame rate
            '-vf': 'transpose=1,hflip'
        })

    # If partial
    if callable(agent_1):
        agent_1 = agent_1()
    if callable(agent_2):
        agent_2 = agent_2()

    # Initialize agents
    if not agent_1.initialized:
        agent_1.get_env_info(env)
    if not agent_2.initialized:
        agent_2.get_env_info(env)
    # 596, 336
    platform1 = env.objects["platform1"]

    for time in tqdm(range(max_timesteps), total=max_timesteps):
        platform1.physics_process(0.05)
        full_action = {
            0: agent_1.predict(obs_1),
            1: agent_2.predict(obs_2)
        }

        observations, rewards, terminated, truncated, info = env.step(
            full_action)
        obs_1 = observations[0]
        obs_2 = observations[1]

        if reward_manager is not None:
            reward_manager.process(env, 1 / env.fps)

        if video_path is not None:
            img = env.render()
            writer.writeFrame(img)
            del img

        if terminated or truncated:
            break

    if video_path is not None:
        writer.close()
    env.close()

    # visualize
    # Video(video_path, embed=True, width=800) if video_path is not None else None
    player_1_stats = env.get_stats(0)
    player_2_stats = env.get_stats(1)

    if player_1_stats.lives_left > player_2_stats.lives_left:
        result = Result.WIN
    elif player_1_stats.lives_left < player_2_stats.lives_left:
        result = Result.LOSS
    else:
        result = Result.DRAW

    match_stats = MatchStats(
        match_time=env.steps / env.fps,
        player1=player_1_stats,
        player2=player_2_stats,
        player1_result=result
    )

    del env

    return match_stats


class ConstantAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = np.zeros_like(self.action_space.sample())
        return action


class RandomAgent(Agent):

    def __init(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.action_space.sample()
        return action


class BasedAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action


class UserInputAgent(Agent):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        # if keys[pygame.K_q]:
        #    action = self.act_helper.press_keys(['q'], action)
        # if keys[pygame.K_v]:
        #    action = self.act_helper.press_keys(['v'], action)
        return action


class ClockworkAgent(Agent):

    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (30, []),
                (7, ['d']),
                (1, ['a']),
                (4, ['a', 'l']),
                (1, ['a']),
                (4, ['a', 'l']),
                (1, ['a']),
                (4, ['a', 'l']),
                (1, ['a']),
                (4, ['a', 'l']),
                (20, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
                (15, ['space']),
                (5, []),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)

        self.steps += 1  # Increment step counter
        return action


class SB3Agent(Agent):

    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    # def set_ignore_grad(self) -> None:
        # self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, callback=None, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )


class TrainLogging(Enum):
    NONE = auto()
    PLOT = auto()
    VIDEO = auto()


def train(agent: Agent,
          reward_manager: RewardManager,
          opponent_cfg: OpponentsCfg = OpponentsCfg(),
          resolution: CameraResolution = CameraResolution.LOW,
          train_timesteps: int = 400_000,
          train_logging: TrainLogging = TrainLogging.PLOT,
          n_envs: int = 1,
          log_dir: str = "/tmp/gym/",
          callback: Optional[Callable] = None,
          log_interval: int = 1
          ):
    # Create environment

    def make_env(rank: int, seed: int = 0):
        def _init():
            env_kwargs = dict(
                reward_manager=reward_manager,
                opponent_cfg=opponent_cfg,
                resolution=resolution,
                rank=rank
            )
            env = SelfPlayWarehouseBrawl(**env_kwargs)
            env.reset(seed=seed + rank)
            # Only wrap with Monitor if we are in a parallel env, otherwise the outer Monitor will handle it
            return Monitor(env, log_dir if n_envs > 1 else None)
        return _init

    if n_envs > 1:
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        # Create a single environment
        env = SelfPlayWarehouseBrawl(
            reward_manager=reward_manager, opponent_cfg=opponent_cfg, resolution=resolution, rank=0)
        if train_logging != TrainLogging.NONE:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)

    # Initialize agent if it hasn't been already
    if not agent.initialized:
        agent.get_env_info(env)

    try:
        # The agent's get_env_info is called by the callback or before training starts
        agent.learn(env, total_timesteps=train_timesteps,
                    callback=callback, verbose=1)
    except KeyboardInterrupt:
        pass

    if train_logging == TrainLogging.PLOT:
        # Plot results - works with both single and parallel environments
        from environment.agent import plot_results
        plot_results(log_dir, title="Learning Curve")

    env.close()
    del env
