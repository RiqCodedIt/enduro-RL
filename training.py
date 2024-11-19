from agent import DQNAgent, preprocess_env
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.96
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 50
TAU = 0.002
LR = 1e-4

gym.register_envs(ale_py)
env = gym.make("ALE/Enduro-v5")
env = preprocess_env(env)
STATES = env.observation_space.shape
ACTIONS = env.action_space.n

agent = DQNAgent(replace_target_cnt=5000, env=env, state_space=STATES, action_space=ACTIONS, eps_strt=EPS_START, eps_dec=EPS_DECAY, batch_size=BATCH_SIZE, lr=LR)
avg_losses_per_episode = agent.train()

plt.figure(figsize=(10, 6))
plt.plot(avg_losses_per_episode, label="Average Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("DQN Training Loss per Episode")
plt.legend()
plt.grid()
# plt.show()

plt.savefig('Training_Loss.png')