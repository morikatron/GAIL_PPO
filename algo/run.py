import random
import pickle
import sys
import os
from collections import deque


from tqdm import tqdm
import gym
import numpy as np
import tensorflow as tf

from util import set_global_seeds
from ppo import PPO
from discriminator import Discriminator
from config import Config
import logger


class CartPoleWrapper(gym.Wrapper):
    """
    CartPoleに関しては、元の環境の報酬の与え方が「1ステップごとに棒が立っていたら報酬が1与えられる」となっていますが、これを
    「最後まで立っていたら報酬1, その前に倒れたら報酬-1、途中の報酬は0」という形に変えます。
    """
    def __init__(self, env):
        super(CartPoleWrapper, self).__init__(env)
        self.steps = 0
        version = env.spec.id.split('-')[-1]
        if version == 'v0':
            self._max_step = 200
        else:
            self._max_step = 500

    def reset(self):
        obs = self.env.reset()
        self.steps = 0
        return obs

    def step(self, ac):
        self.steps += 1
        obs, rew, done, info = self.env.step(ac)
        if done:
            if self.steps == self._max_step:
                rew = 1
            else:
                rew = -1
        else:
            rew = 0
        return obs, rew, done, info


class ExpertDataset:
    def __init__(self, demo_path):
        self.data_set = self._load_demo(demo_path)
        self.size = len(self.data_set)
        self.idx = 0

    def sample(self, batch_size):
        mini_batch_act = []
        mini_batch_obs = []
        for i in range(batch_size):
            mini_batch_obs.append(self.data_set[self.idx][0])
            mini_batch_act.append(self.data_set[self.idx][1])
            self.idx = (self.idx + 1) % self.size
        mini_batch_obs = tf.convert_to_tensor(mini_batch_obs, dtype=tf.float32)
        mini_batch_act = tf.convert_to_tensor(mini_batch_act, dtype=tf.int32)
        return mini_batch_obs, mini_batch_act

    def _load_demo(self, demo_path):
        data_set = []
        with open(demo_path, "rb") as f:
            trajectories = pickle.load(f)
            for epi in trajectories:
                for obs, action, _, _, _ in epi:
                    data_set.append((obs, action))
        random.shuffle(data_set)
        return data_set


class Memory:
    """
    集めたサンプルを学習用に保存しておくメモリークラスです。
    サンプルを収集するステップ数分のサイズを
    deltaの値を計算する際に1ステップ先の状態価値が必要になるため、valueのサイズは1つ余分に大きくとり」、
    GAEを計算する際に1ステップ先のGAEの値が必要になるため、GAEのサイズを1つ余分に大きくとっています。
    """
    def __init__(self, obs_shape, hparams):
        self._hparams = hparams
        self.size = self._hparams.num_steps
        self.obses   = np.zeros((self.size, )+obs_shape)
        self.actions = np.zeros((self.size, ))
        self.rewards = np.zeros((self.size,   1))
        self.dones   = np.zeros((self.size,   1))
        self.values  = np.zeros((self.size+1, 1))
        self.policy  = np.zeros((self.size,   1))
        self.deltas  = np.zeros((self.size,   1))
        self.discounted_rew_sum = np.zeros((self.size, 1))
        self.gae = np.zeros((self.size+1, 1))
        self.sample_i = 0  # サンプルをメモリに保存するためのポインタの役割を果たします

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, done, value, policy):
        """
        サンプルをメモリに保存する時点では、 1ステップ先の状態価値はまだ不明なのでdeltaのみ1ステップ分遅延させて保存します。
        その都合上メモリのサイズより1回分多くaddが呼ばれますが、最後のaddではdeltaのみ保存するようにします。
        """
        if self.sample_i < len(self.obses):
            self.obses[self.sample_i] = obs
            self.actions[self.sample_i] = action
            self.rewards[self.sample_i] = reward
            self.dones[self.sample_i] = done
            self.values[self.sample_i] = value
            self.policy[self.sample_i] = policy
        else:
            self.values[self.sample_i] = value
        self.sample_i += 1

    def compute_gae(self):
        """
        Generalized Advantage Estimatorを後ろからさかのぼって計算します。
        最後の状態のGAEはdeltaと等しく、それ以前は次の状態のgaeをgamma * lambdaで割り引いた値にdeltaの値を足したものになります。
        """
        # ステップtでエピソードが終了していた場合、V(t+1)は次のエピソードの最初の状態価値なのでdeltaを計算する際にエピソードをまたがないようにV(t+1)は0とします。(エピソード終了後に得られる報酬は0なので状態価値も0です。)
        for i in reversed(range(self.size+1)):
            self.deltas[i - 1] = self.rewards[i - 1] + self._hparams.gamma * self.values[i] * (1 - self.dones[i - 1]) - self.values[i - 1]
        self.gae[-1] = self.deltas[-1]
        for t in reversed(range(self.size-1)):
            self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (self._hparams.gamma * self._hparams.lambda_) * self.gae[t + 1]
        self.discounted_rew_sum = self.gae[:-1] + self.values[:-1]
        self.gae = (self.gae - np.mean(self.gae[:-1])) / (np.std(self.gae[:-1]) + 1e-8)  # 正規化をしておきます。
        return

    def sample(self, idxes):
        batch_obs = tf.convert_to_tensor(self.obses[idxes], dtype=tf.float32)
        batch_act = tf.convert_to_tensor(self.actions[idxes], dtype=tf.int32)
        batch_adv = tf.squeeze(tf.convert_to_tensor(self.gae[idxes], dtype=tf.float32))
        batch_pi = tf.squeeze(tf.convert_to_tensor(self.policy[idxes], dtype=tf.float32))
        batch_sum = tf.squeeze(tf.convert_to_tensor(self.discounted_rew_sum[idxes], dtype=tf.float32))
        return batch_obs, batch_act, batch_adv, batch_sum, batch_pi

    def reset(self):
        self.sample_i = 0
        self.obses = np.zeros_like(self.obses)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.values = np.zeros_like(self.values)
        self.policy = np.zeros_like(self.policy)
        self.deltas = np.zeros_like(self.deltas)
        self.discounted_rew_sum = np.zeros_like(self.discounted_rew_sum)
        self.gae = np.zeros_like(self.gae)


def main(args):
    config = Config()
    set_global_seeds(config.seed)
    logger.configure("logs")
    env = gym.make(config.env_name)
    env = CartPoleWrapper(env)
    # with tf.device("/gpu:0"):  # gpuを使用する場合
    with tf.device("/cpu:0"):
        ppo = PPO(
            num_actions=env.action_space.n,
            input_shape=env.observation_space.shape,
            config=config
        )
        discriminator = Discriminator(
            num_obs=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            config=config
        )
    num_episodes = 0
    episode_rewards = deque([], maxlen=100)
    memory = Memory(env.observation_space.shape, config)
    # ----- load trajectories -----
    if len(args) > 1 and args[1] == "ppo":
        print("use ppo demonstrations")
        demo_path = f"demo/{env.spec.id}_ppo.pkl"
        if not os.path.exists(demo_path):
            import generate_demo
            print("generating demonstrations...")
            generate_demo.main()
        expert_dataset = ExpertDataset(f"demo/{env.spec.id}_ppo.pkl")
    else:
        print("use human demonstrations")
        demo_path = f"demo/{config.demo}"
        assert os.path.exists(demo_path), f"demonstrations {demo_path} not found. execute make_demo.py"
        expert_dataset = ExpertDataset(demo_path)
    reward_sum = 0
    obs = env.reset()
    for t in tqdm(range(config.num_updates)):
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

        # ===== 2. train reward giver(discriminator) =====
        for _ in range(config.num_discriminator_epochs):
            idxes = [idx for idx in range(config.num_steps)]
            random.shuffle(idxes)
            for start in range(0, len(memory), config.batch_size):
                minibatch_indexes = idxes[start:start + config.batch_size]
                agent_obs, agent_act, _, _, _ = memory.sample(minibatch_indexes)
                demo_obs, demo_act = expert_dataset.sample(config.batch_size)
                total_loss, agent_loss, demo_loss, agent_acc, demo_acc = discriminator.train(demo_obs, demo_act,
                                                                                             agent_obs, agent_act)
            actions = tf.constant(memory.actions, dtype=tf.int32)
            observations = tf.constant(memory.obses, dtype=tf.float32)
            reward_signals = discriminator.inference(observations, actions).numpy()
            memory.rewards = reward_signals
            rew_mean = np.mean(reward_signals)

        if t % config.log_step == 0:
            logger.record_tabular("discriminator loss total", total_loss.numpy())
            logger.record_tabular("discriminator loss agent", agent_loss.numpy())
            logger.record_tabular("discriminator loss demo", demo_loss.numpy())
            logger.record_tabular("reward signal mean", rew_mean)
            logger.record_tabular("agent acc", agent_acc.numpy())
            logger.record_tabular("demo acc", demo_acc.numpy())

        # ===== train agent(generator) =====
        memory.compute_gae()
        for _ in range(config.num_generator_epochs):
            idxes = [idx for idx in range(config.num_steps)]
            random.shuffle(idxes)
            for start in range(0, len(memory), config.batch_size):
                minibatch_indexes = idxes[start:start+config.batch_size]
                batch_obs, batch_act, batch_adv, batch_sum, batch_pi_old = memory.sample(minibatch_indexes)
                loss, policy_loss, value_loss, entropy_loss, policy, kl, frac = ppo.train(batch_obs, batch_act, batch_pi_old, batch_adv, batch_sum)

        if t % config.log_step == 0:
            logger.record_tabular("num episodes", num_episodes)
            logger.record_tabular("loss", loss.numpy())
            logger.record_tabular("policy loss", policy_loss.numpy())
            logger.record_tabular("value loss", value_loss.numpy())
            logger.record_tabular("entropy loss", entropy_loss.numpy())
            logger.record_tabular("kl", kl.numpy())
            logger.record_tabular("frac", frac.numpy())
            logger.record_tabular("mean 100 episode reward", np.mean(episode_rewards))
            logger.record_tabular("max 100 episode reward", np.max(episode_rewards))
            logger.record_tabular("min 100 episode reward", np.min(episode_rewards))
            logger.dump_tabular()
        memory.reset()
    # ===== finish training =====
    if config.play:
        obs = env.reset()
        while True:
            action, _ = ppo.step(obs[np.newaxis, :])
            action = int(action.numpy()[0])
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == "__main__":
    main(sys.argv)
