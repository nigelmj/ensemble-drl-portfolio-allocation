"""Stable-Baselines3 callbacks used by the portfolio agents."""

from __future__ import annotations

import statistics

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except Exception:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except Exception:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
        return True

    def _on_rollout_end(self) -> bool:
        # rollout_buffer only exists for on-policy algos (PPO/A2C); off-policy
        # algos (e.g. SAC) skip these records silently.
        if "rollout_buffer" not in self.locals:
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            return True
        try:
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except Exception:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
        return True
