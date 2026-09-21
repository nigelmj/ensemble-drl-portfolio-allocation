# DRL agent specialized for the PortfolioAllocationEnv using Stable Baselines 3
from __future__ import annotations

from typing import Type

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from rl_portfolio.utils.callbacks import TensorboardCallback

MODELS = {"ppo": PPO, "a2c": A2C, "sac": SAC}

MODEL_KWARGS = {
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.001,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "a2c": {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.0007,
    },
    "sac": {
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
}

NOISE = {"normal": NormalActionNoise}


class PortfolioAllocationDRLAgent:
    """Provides implementations for DRL algorithms
    specialized for the PortfolioAllocationEnv.

    Attributes
    ----------
        env: PortfolioAllocationEnv
            user-defined portfolio allocation environment

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=42,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name].copy()

        model_kwargs = dict(model_kwargs)
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        tb_log_name,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):  # this function is static method, so it can be called without creating an instance of the class
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """make a prediction and get results"""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        episode_stats = None

        test_env.reset()
        max_steps = environment.episode_length - 1

        for i in range(max_steps + 1):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)

            if i == max_steps - 1:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
                episode_stats = test_env.env_method(method_name="get_episode_stats")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0], episode_stats[0]
