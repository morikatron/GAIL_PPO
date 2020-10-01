import tensorflow as tf
EPS = 1e-8


class Discriminator(tf.Module):
    """
    ニューラルネットワークが絡む推論、学習の２つの計算を行うクラスです。
    @tf.functionデコレータは実行時に関数をグラフにコンパイルしてくれるもので、学習の高速化・GPUでの実行を可能にします。
    """
    def __init__(self, num_obs, num_actions, config):
        super(Discriminator, self).__init__(name='discriminator')
        self.num_actions = num_actions
        self.input_shape = (num_obs + num_actions, )
        self.batch_size = config.batch_size
        self.config = config

        self.network = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.gradient_clip = config.gradient_clip

    @tf.function
    def inference(self, obs, action):
        one_hot_action = tf.one_hot(action, self.num_actions, dtype=tf.float32)
        agent_input = tf.concat([obs, one_hot_action], axis=1)
        p = self.network(agent_input)
        # reward = tf.math.log(p + EPS)
        reward = p
        return reward

    @tf.function
    def train(self, demo_obs, demo_act, agent_obs, agent_act):
        with tf.GradientTape() as tape:
            agent_one_hot_act = tf.one_hot(agent_act, self.num_actions)
            demo_one_hot_act = tf.one_hot(demo_act, self.num_actions)
            agent_input = tf.concat([agent_obs, agent_one_hot_act], axis=1)
            demo_input = tf.concat([demo_obs, demo_one_hot_act], axis=1)
            agent_p = self.network(agent_input)
            demo_p = self.network(demo_input)
            agent_loss = - tf.reduce_mean(tf.math.log(1. - agent_p + EPS))
            demo_loss = - tf.reduce_mean(tf.math.log(demo_p + EPS))
            loss = agent_loss + demo_loss
        grads = tape.gradient(loss, self.network.trainable_variables)
        if self.gradient_clip is not None:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.gradient_clip))
            grads = clipped_grads
        agent_acc = tf.reduce_mean(tf.cast(agent_p < 0.5, dtype=tf.float32))
        demo_acc = tf.reduce_mean(tf.cast(demo_p > 0.5, dtype=tf.float32))
        grads_and_vars = zip(grads, self.network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return loss, agent_loss, demo_loss, agent_acc, demo_acc

    def _build_model(self):
        input_x = tf.keras.Input(shape=self.input_shape)
        h1 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(input_x)
        h2 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(h1)
        p = tf.keras.layers.Dense(1, activation="sigmoid")(h2)
        model = tf.keras.Model(inputs=input_x, outputs=p)
        return model


class SoftMaxDiscriminator(tf.Module):
    """
    ニューラルネットワークが絡む推論、学習の２つの計算を行うクラスです。
    @tf.functionデコレータは実行時に関数をグラフにコンパイルしてくれるもので、学習の高速化・GPUでの実行を可能にします。
    """

    def __init__(self, obs_shape, num_actions, config):
        super(SoftMaxDiscriminator, self).__init__(name='sof_tmax_discriminator')
        self.num_actions = num_actions
        self.input_shape = obs_shape
        self.batch_size = config.batch_size
        self.config = config

        self.network = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.gradient_clip = config.gradient_clip

    @tf.function
    def inference(self, obs, action):
        p = self.network(obs)
        one_hot_action = tf.one_hot(action, self.num_actions, dtype=tf.float32)
        reward = tf.reduce_sum(tf.multiply(p, one_hot_action))
        # reward = tf.math.log(p + EPS)
        return reward

    @tf.function
    def train(self, demo_obs, demo_act, agent_obs, agent_act):
        with tf.GradientTape() as tape:
            agent_one_hot_act = tf.one_hot(agent_act, self.num_actions)
            demo_one_hot_act = tf.one_hot(demo_act, self.num_actions)
            agent_p = self.network(agent_obs)
            demo_p = self.network(demo_obs)
            agent_loss = - tf.reduce_mean(tf.math.log(1. - agent_p + EPS))
            demo_loss = - tf.reduce_mean(tf.math.log(demo_p + EPS))
            loss = agent_loss + demo_loss
        grads = tape.gradient(loss, self.network.trainable_variables)
        if self.gradient_clip is not None:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.gradient_clip))
            grads = clipped_grads
        agent_acc = tf.reduce_mean(tf.cast(agent_p < 0.5, dtype=tf.float32))
        demo_acc = tf.reduce_mean(tf.cast(demo_p > 0.5, dtype=tf.float32))
        grads_and_vars = zip(grads, self.network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return loss, agent_loss, demo_loss, agent_acc, demo_acc

    def _build_model(self):
        input_x = tf.keras.Input(shape=self.input_shape)
        h1 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(input_x)
        h2 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(h1)
        p = tf.keras.layers.Dense(1, activation="sigmoid")(h2)
        model = tf.keras.Model(inputs=input_x, outputs=p)
        return model