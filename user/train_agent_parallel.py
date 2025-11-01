'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

from user.reward_agents import *
from environment.agent_parallel import *
from user.callbacks import SaveAndRenderCallback
import torch
from torch.nn import functional as F
from torch import nn as nn
import gymnasium as gym
import pygame

from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList

# TODO: Fix name conflicts
# from environment.agent import *
from environment.agent_parallel import Agent
from environment.agent import JumpingAgent
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------


class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''

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

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )


class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''

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
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''

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

        return action


class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''

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


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''

    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return {
            "features_extractor_class": cls,
            "features_extractor_kwargs": {"features_dim": features_dim, "hidden_dim": hidden_dim},
        }


class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(
            ), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
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

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# -------------------------------------------------------------------------
# -------------------------- Curriculum Training --------------------------
# -------------------------------------------------------------------------


def train_stage(my_agent, save_name, stage_number, total_iterations):
    match stage_number:
        case 0:
            reward_manager = BasicMovementCurriculum()
            save_freq = 405000
            stage_iterations = 2025000
        case 1:
            reward_manager = StopFallingCurriculum()
            save_freq = 405000
            stage_iterations = 2025000
        case 2:
            reward_manager = TowardsOpponentCurriculum()
            save_freq = 405000
            stage_iterations = 8025000
            opponent_specification = {
                'constant_agent': (1, partial(ConstantAgent))
            }
        case 3:
            reward_manager = KillOpponentCurriculum()
            save_freq = 202500
            stage_iterations = save_freq * 5
            opponent_specification = {
                'constant_agent': (0.5, partial(ConstantAgent)),
                'jumping_agent': (0.5, partial(JumpingAgent)),
            }

    save_path = 'checkpoints_parallel'
    run_name = f'{save_name}_stage_{stage_number}'
    log_dir = os.path.join(save_path, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Set opponent settings here:
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    # Setup callback
    callback = SaveAndRenderCallback(
        save_freq=save_freq,
        save_path=save_path,
        run_name=run_name,
        # Render every save (every 405k steps instead of every 2M)
        render_freq=save_freq,
        reward_manager=reward_manager,
        resolution=CameraResolution.LOW,
        agent_class=type(my_agent),
        sb3_class=my_agent.sb3_class,
        opponent_cfg=opponent_cfg,
        verbose=1
    )

    train(
        agent=my_agent,
        reward_manager=reward_manager,
        opponent_cfg=opponent_cfg,
        resolution=CameraResolution.LOW,
        train_timesteps=stage_iterations,
        train_logging=TrainLogging.PLOT,
        n_envs=16,
        log_dir=log_dir,
        callback=callback,
        log_interval=10  # Log every 10 updates
    )
    total_iterations += stage_iterations

    return


def train_basic_curriculum(start_stage: int = 0, file_path: str | None = None):
    if file_path is not None:
        my_agent = RecurrentPPOAgent(file_path=file_path)
        my_agent.get_env_info(SelfPlayWarehouseBrawl())
    else:
        my_agent = RecurrentPPOAgent()

    save_name = "everything_16_agent"

    # Stage 1
    total_iterations = 0
    for stage_number in range(start_stage, 4):
        train_stage(my_agent, save_name, stage_number, total_iterations)


# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''


if __name__ == '__main__':
    # Create agent
    # my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Resume training from latest checkpoint
    train_basic_curriculum(
        start_stage=3,
        file_path="checkpoints_parallel/everything_16_agent_stage_2/rl_model_7290000_steps.zip"
    )
