from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match, run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

# my_agent = UserInputAgent()
# my_agent = ConstantAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
# file_path = "checkpoints/experiment_#/file_name"
my_agent = SubmittedAgent(
    file_path="checkpoints/curriculum_two_stage_v3_stage_0/rl_model_405000_steps.zip")
opponent = SubmittedAgent(
    file_path='checkpoints/curriculum_1_v2/rl_model_202500_steps.zip'
)

match_time = 30

from user.reward_agents import StopFallingCurriculum

# Run a single real-time match
# use run_real_time_match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='tt_agent.mp4',
    reward_manager=StopFallingCurriculum()
)
# run_real_time_match(
#     agent_1=my_agent,
#     agent_2=opponent,
#     max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
#     resolution=CameraResolution.LOW
# )