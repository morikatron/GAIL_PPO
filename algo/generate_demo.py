import random
import pickle
from collections import deque
import os

import gym
import numpy as np
import tensorflow as tf

import logger
from run import Memory, CartPoleWrapper
from config import Config
from util import set_global_seeds
from ppo import PPO


def main():
    """
    generate expert trajectory using ppo
    """
    config = Config()
    set_global_seeds(config.seed)
    env = gym.make(config.env_name)
    env = CartPoleWrapper(env)
    # with tf.device("/gpu:0"):  # gpuを使用する場合
    with tf.device("/cpu:0"):
        ppo = PPO(
            num_actions=env.action_space.n,
            input_shape=env.observation_space.shape,
            config=config
        )
    num_episodes = 0
    episode_rewards = deque([], maxlen=100)
    memory = Memory(env.observation_space.shape, config)
    reward_sum = 0
    obs = env.reset()
    while True:
        # ===== 1. get samples =====
        for _ in range(config.num_steps):
            policy, value = ppo.step(tf.constant(obs))
            policy = policy.numpy()
            action = np.random.choice(2, p=policy)
            next_obs, rew, done, _ = env.step(action)
            # rew = discriminator.step(tf.constant(obs, dtype=tf.float32), tf.constant(action, dtype=tf.int32))
            memory.add(obs, action, rew, done, value, policy[action])
            obs = next_obs
            reward_sum += rew
            if done:
                episode_rewards.append(env.steps)
                num_episodes += 1
                reward_sum = 0
                obs = env.reset()
        _, last_value = ppo.step(obs[np.newaxis, :])
        memory.add(None, None, None, None, last_value, None)

        # ===== train agent =====
        memory.compute_gae()
        for _ in range(config.num_generator_epochs):
            idxes = [idx for idx in range(config.num_steps)]
            random.shuffle(idxes)
            for start in range(0, len(memory), config.batch_size):
                minibatch_indexes = idxes[start:start+config.batch_size]
                batch_obs, batch_act, batch_adv, batch_sum, batch_pi_old = memory.sample(minibatch_indexes)
                loss, policy_loss, value_loss, entropy_loss, policy, kl, frac = ppo.train(batch_obs, batch_act, batch_pi_old, batch_adv, batch_sum)
        print(f"\r 100 epi mean: {np.mean(episode_rewards)}", end="")
        memory.reset()
        if np.mean(episode_rewards) >= 180:
            print("training finished!")
            break
    # ===== finish training =====
    trajectories = []
    while True:
        obs = env.reset()
        episode_trajectory = []
        done = False
        while not done:
            prev_obs = obs
            policy, _ = ppo.step(tf.constant(obs))
            action = np.random.choice(2, p=policy.numpy())
            obs, rew, done, _ = env.step(action)
            episode_trajectory.append((np.array(prev_obs), np.array(action, dtype='int64'), rew, np.array(obs), done))
        if len(episode_trajectory) >= 200:
            trajectories.append(episode_trajectory)
            print("saved trajectory len", len(episode_trajectory))
        else:
            print("reset! episode reward is", len(episode_trajectory))
        if len(trajectories) >= 10:
            break
    if not os.path.exists("../demo"):
        os.mkdir("demo")
    with open(f"demo/{env.spec.id}_ppo.pkl", mode="wb") as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    main()
