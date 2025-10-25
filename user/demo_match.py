from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match, run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

# my_agent = UserInputAgent()
my_agent = ConstantAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
# file_path = "checkpoints/experiment_#/file_name"
opponent = SubmittedAgent(
    file_path="checkpoints/death_penalty_air_penalty/rl_model_108000_steps.zip")

match_time = 30

# Run a single real-time match
# use run_real_time_match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='tt_agent.mp4'
)
# run_real_time_match(
#     agent_1=my_agent,
#     agent_2=opponent,
#     max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
#     resolution=CameraResolution.LOW
# )