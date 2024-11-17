import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from model import DQN
from memory import ReplayMemory
import random
import math
from PIL import Image
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def preprocess_env(env):
    # env = AtariPreprocessing(env, screen_size=(84, 84),frame_skip=1, grayscale_obs=True)
    env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    return env
# from transforms import Transforms

class DQNAgent(object):
    def __init__(self, replace_target_cnt, env, state_space, action_space, tau=0.005, 
                 model_name='enduro_model', gamma=0.99, eps_strt=0.1, 
                 eps_end=0.001, eps_dec=5e-6, batch_size=128, lr=0.001):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.GAMMA = gamma
        self.LR = lr
        self.eps = eps_strt
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.tau = tau

        #Use GPU if available
        if torch.backends.mps.is_available():
            print('MPS FOUND')
            self.device = torch.device("mps")
        else:
            print ("MPS device not found.")
            self.device = torch.device("cpu")

        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        #initialize ReplayMemory
        self.memory = ReplayMemory(10000)

        # After how many training iterations the target network should update
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        self.policy_net = DQN(self.state_space, self.action_space, filename=model_name).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename=model_name+'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model('/Users/tariqgeorges/Documents/Riq Coding/Nov 24/game-rl/models/enduro_modelNONE.pth')
            print('loaded pretrained model')
        except:
            print('Didnt load model')
            pass

         # Set target net to be the same as policy net
        self.replace_target_net()

        #Set optimizer & loss
        self.optim = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.loss = torch.nn.SmoothL1Loss()
    
    def sample_batch(self):
        batch = self.memory.sample(self.batch_size)
        # print(f'Batch.state = {batch.state}')
        # print(f'Batch.state[0].shape = {batch.state[0].shape}')
        state_shape = batch.state[0].shape
        #Batch.state[0].shape (4, 84, 84)

        # Convert to tensors with correct dimensions
        state = torch.tensor(batch.state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        state_ = torch.tensor(batch.next_state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done


        #------------------------------------------------#
        # # Unpack batch of transitions into separate lists for each attribute
        # states, actions, rewards, next_states, dones = zip(*batch)

        # print(f'States {states}, actions {actions}, rewards {rewards}, next_states{next_states}')

        # Convert each list to a tensor with the correct dimensions
        # state = torch.tensor(batch.state).float().to(self.device)
        # action = torch.tensor(batch.action).unsqueeze(1).to(self.device)  # Ensure actions are 2D (batch_size, 1)
        # reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)  # Ensure rewards are 2D (batch_size, 1)
        # state_ = torch.tensor(batch.next_state).float().to(self.device)
        # done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)  # Ensure dones are 2D (batch_size, 1)


        # return state, action, reward, state_, done

    # Returns the greedy action according to the policy net
    
    def greedy_action(self, obs):
    # Ensure obs_ is just the raw observation array
        if isinstance(obs, tuple):
            obs = obs[0]  # If step returns a tuple, get the observation
        # print("Choosing Greedy Action")
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    
    def choose_action(self, obs):
        # print("Choosing Action")
        if random.random() > self.eps:
            action = self.greedy_action(obs)
        else:
            action = random.choice([x for x in range(self.action_space)])
        return action
    
    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)

            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('Target network replaced')
    
    # Decrement epsilon 
    def dec_eps(self):
        self.eps = max(self.eps_end, self.eps - (self.eps - self.eps_end) / self.eps_dec)
        
    def play_games(self, num_eps, render=True):
        # Set network to eval mode
        self.policy_net.eval()
        self.env = gym.make("Enduro-v4", render_mode="human")
        self.env = preprocess_env(self.env)

        scores = []

        for i in range(num_eps):
            done = False

            # Get preprocessed observation from environment
            state, _ = self.env.reset()
            
            score = 0
            cnt = 0
            while not done:
                # Take the greedy action and observe next state
                action = self.greedy_action(state)
                next_state, reward, done, _, info = self.env.step(action)
                if render:
                    self.env.render()


                # Store transition
                self.memory.push(state, action, next_state, reward, int(done))

                # Calculate score, set next state and obs, and increment counter
                score += reward
                state = next_state
                cnt += 1

            # If the score is more than 300, save a gif of that game
            if score > 300:
                self.save_gif(cnt)

            scores.append(score)
            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                    \n\tEpsilon: {self.eps}\n\tSteps made: {cnt}')
        
        self.env.close()

    def learn(self, num_iters=1):
        # print('Learning Func')
        # Skip learning if there's not enough memory
        if self.memory.pointer < self.batch_size:
            return 
        
        losses = []

        for i in range(num_iters):
            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the Q-value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate the best next action value from the target net and detach it from the computation graph
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)

            # Calculate the target Q-value
            # (1 - done) ensures q_target is 0 if transition is in a terminal state
            q_target = reward + (1 - done) * (self.GAMMA * q_next)

            # Compute the loss
            loss = self.loss(q_eval, q_target).to(self.device)

            losses.append(loss.cpu().item())

            # Perform backward propagation and optimization step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Increment learn_counter (used for epsilon decay and target network updates)
            self.learn_counter += 1

            # Check if it's time to replace the target network
            self.replace_target_net()

        # Save the model and decrement epsilon
        self.policy_net.save_model()
        self.dec_eps()

        return losses

    def save_gif(self, num_transitions):
        frames = []
        for i in range(self.memory.pointer - num_transitions, self.memory.pointer):
            frame = Image.fromarray(self.memory.memory[i].raw_state, mode='RGB')
            frames.append(frame)
        
        frames[0].save('episode.gif', format='GIF', append_images=frames[1:], save_all=True, duration=10, loop=0)
    
    # Plays num_eps amount of games, while optimizing the model after each episode
    def train(self, num_eps=450, render=False):
        scores = []
        avg_losses_per_episode = []
        max_score = 0

        for i in range(num_eps):
            done = False
            max_steps = 8000
            # Reset environment and preprocess state
            state, _ = self.env.reset()
            # print(f"AFTER RESET, STATE SHAPE IS {state.shape}")
            
            score = 0
            cnt = 0
            while not done and cnt < max_steps:
                # Take epsilon greedy action
                action = self.choose_action(state)
                next_state, reward, done, _, info = self.env.step(action)
                if render:
                    self.env.render()

                # print(f"AFTER RESET, NEXT STATE IS {obs_.shape}")

                # Preprocess next state and store transition
                self.memory.push(state, action, reward, next_state, int(done))

                score += reward
                state = next_state
                cnt += 1

            # Maintain record of the max score achieved so far
            if score > max_score:
                max_score = score

            # Save a gif if episode is best so far
            if score > 300 and score >= max_score:
                self.save_gif(cnt)

            scores.append(score)
            # print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
            #     \n\tEpsilon: {self.eps}\n\tTransitions added: {cnt}')
            
            # Train on as many transitions as there have been added in the episode
            print(f'Learning x{math.ceil(cnt/self.batch_size)}')
            eps_losses = self.learn(math.ceil(cnt/self.batch_size))


            avg_loss = np.mean(eps_losses) if eps_losses else 0
            avg_losses_per_episode.append(avg_loss)

            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg loss: {avg_loss:.4f}\
            \n\tAvg score (past 100): {np.mean(scores[-100:])}\
            \n\tEpsilon: {self.eps}\n\tTransitions added: {cnt}')

        self.env.close()
        return avg_losses_per_episode
    
