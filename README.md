# GAIL_PPO
Generative Adversarial Imitation Learning implementation with Tensorflow2  
This implementation only supports CartPole environment(OpenAI gym).  


このリポジトリは逆強化学習型模倣学習アルゴリズムGenerative Adversarial Imitation LearningをTensorflow2で実装したものです。  
学習環境についてはCartPole-v0でのみ検証済みです。  
GAILについて解説したブログは[こちら](https://tech.morikatron.ai/entry/2020/10/12/100000)になります。  

## Relevant Papers
 - Generative Adversarial Imitation Learning, Jonathan Ho, Stefano Ermon, 2016  
https://arxiv.org/abs/1606.03476  
 - Proximal Policy Optimization Algorithms, Schulman et al. 2017  
https://arxiv.org/abs/1707.06347  


## Requirements
 - Python3
 - tensorflow2
 - gym
 - tqdm

## Usage
  - clone this repo
 ```
 $ git clone https://github.com/morikatron/GAIL_PPO.git
 ```
  - change directory and run(using human demonstrations) 
 ```
 $ cd GAIL_PPO
 $ python algo/run.py
 ```
  - run(using ppo demonstrations)
 ```
 $ python algo/run.py ppo
 ```
   - make demo
 ```
 $ python algo/make_demo.py  # human demonstrations
 $ python algo/generate_demo.py  # ppo demonstrations
 ```
 ## Performance Example
 ![using human demonstratioins](https://github.com/morikatron/GAIL_PPO/blob/master/sample_result/human/plot.png)
 ![using ppo demonstrations](https://github.com/morikatron/GAIL_PPO/blob/master/sample_result/ppo/plot.png)

