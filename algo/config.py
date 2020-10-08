from typing import NamedTuple


class Config(NamedTuple):
    # learning config
    env_name: str = "CartPole-v0"  # OpenAI gym env id
    demo: str = "CartPole-v0_human.pkl"
    seed: int = 1234  # random seed
    num_updates: int = 10000  # total training iterations
    log_step: int = 100  # log frequency
    play: bool = True  # render after training
    # hyper-parameters
    batch_size: int = 32  # batch size
    num_generator_epochs: int = 3  # number of epochs for training ppo policy
    num_discriminator_epochs: int = 3  # number of epochs for training discriminator
    num_steps: int = 128  # horizon
    num_units: int = 64  # fc units
    gamma: float = 0.99  # discount rate
    lambda_: float = 0.95  # gae discount rate
    clip: float = 0.2  # clipping c
    vf_coef: float = 0.5  # coefficient of value loss
    ent_coef: float = 0.01  # coefficient of entropy
    learning_rate: float = 2.5e-4  # learning rate
    gradient_clip: float = 0.5  # gradinet clipping