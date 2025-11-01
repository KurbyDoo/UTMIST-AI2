import os
import tempfile
import csv
import time
from stable_baselines3.common.callbacks import BaseCallback
from environment.agent_parallel import run_match, CameraResolution, RewardManager, Agent, OpponentsCfg
from typing import Type
from stable_baselines3.common.base_class import BaseAlgorithm


class SaveAndRenderCallback(BaseCallback):
    """
    A custom callback that saves the model and renders a demo video periodically.
    This is designed to work with parallel environments (VecEnv).
    """

    def __init__(self, save_freq: int, save_path: str, run_name: str,
                 render_freq: int, reward_manager: RewardManager, resolution: CameraResolution,
                 agent_class: Type[Agent], sb3_class: Type[BaseAlgorithm], opponent_cfg: OpponentsCfg,
                 name_prefix: str = "rl_model", max_saved: int = 20, verbose: int = 0):
        super(SaveAndRenderCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.render_freq = render_freq
        self.save_path = save_path
        self.run_name = run_name
        self.name_prefix = name_prefix
        self.max_saved = max_saved
        self.reward_manager = reward_manager
        self.resolution = resolution
        self.agent_class = agent_class
        self.sb3_class = sb3_class
        self.opponent_cfg = opponent_cfg

        self.experiment_path = os.path.join(self.save_path, self.run_name)
        os.makedirs(self.experiment_path, exist_ok=True)

        self.saved_models = []
        self.games_completed = 0
        self.start_time = time.time()
        self.first_step_done = False  # Track if we've rendered initial demo

        # Setup CSV logging
        self.log_file = os.path.join(self.experiment_path, "training_log.csv")
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ['timesteps', 'games_completed', 'time_elapsed', 'fps'])

        if self.verbose > 0:
            print(f"Logging training metrics to {self.log_file}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Render initial demo video on first step (when resuming training)
        if not self.first_step_done and self.n_calls == 1:
            self.first_step_done = True
            self._render_demo_video(suffix="_initial")

        # Check for finished episodes
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # an episode is done
                info = self.locals.get("infos", [{}])[i]
                if "episode" in info:
                    self.games_completed += 1

        self.logger.record('custom/games_completed', self.games_completed)

        # Write to CSV log every 10000 steps
        if self.n_calls % 10000 == 0:
            time_elapsed = time.time() - self.start_time
            fps = self.num_timesteps / time_elapsed if time_elapsed > 0 else 0
            self.csv_writer.writerow(
                [self.num_timesteps, self.games_completed, f'{time_elapsed:.2f}', f'{fps:.2f}'])
            self.csv_file.flush()
            if self.verbose > 0:
                print(
                    f"[LOG] Timesteps: {self.num_timesteps}, Games: {self.games_completed}, FPS: {fps:.2f}")

        # Save model based on num_timesteps (not n_calls) to work correctly with parallel envs
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            path = os.path.join(
                self.experiment_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model to {path}")
            self.saved_models.append(path)
            if self.max_saved > 0 and len(self.saved_models) > self.max_saved:
                os.remove(self.saved_models.pop(0))

        # Render demo videos based on num_timesteps
        if self.render_freq > 0 and self.num_timesteps > 0 and self.num_timesteps % self.render_freq == 0:
            self._render_demo_video()

        return True

    def _render_demo_video(self, suffix: str = "") -> None:
        """
        Render a demo video of the current agent playing against an opponent.

        Args:
            suffix: Optional suffix to add to the video filename (e.g., "_initial")
        """
        if suffix:
            video_path = os.path.join(
                self.experiment_path, f"demo_game_t{self.num_timesteps}{suffix}.mp4")
        else:
            video_path = os.path.join(
                self.experiment_path, f"demo_game_t{self.num_timesteps}.mp4")

        if self.verbose > 0:
            print(f"Rendering demo video to {video_path}")

        # Create a temporary agent from the current model to pass to run_match
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_model_path = os.path.join(tmpdirname, "temp_model.zip")
            self.model.save(temp_model_path)

            temp_agent = self.agent_class(file_path=temp_model_path)

            # Create a dummy env to initialize the agents
            from environment.agent_parallel import SelfPlayWarehouseBrawl
            dummy_env = SelfPlayWarehouseBrawl()
            temp_agent.get_env_info(dummy_env)

            # Get an opponent from the opponent config
            self.opponent_cfg.env = dummy_env
            opponent_agent = self.opponent_cfg.on_env_reset()

            dummy_env.close()

            run_match(
                agent_1=temp_agent,
                agent_2=opponent_agent,
                max_timesteps=30 * 90,  # 90 second match (2700 frames)
                video_path=video_path,
                resolution=self.resolution,
                reward_manager=self.reward_manager
            )
