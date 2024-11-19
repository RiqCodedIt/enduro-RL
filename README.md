# Enduro Deep Reinforcment Learning
This project is about using Deep Q learning to play the atari game Enduro
<!-- 
<p align="center">
  <img  src="./lunar-lander.gif">
</p> -->

# These points are important read them please:
- Best model will be uploaded to git

- Loss is plotted per episode after training

- requirements.txt is included

Step-by-Step Workflow
1. Set Hyperparameters:
Adjust learning parameters such as:
batch_size: Number of transitions per training step.
lr: Learning rate for the optimizer.
eps_start, eps_end, eps_dec: Parameters for epsilon-greedy exploration.
replace_target_cnt: Frequency of target network updates.
2. Run Experiments:
Test different exploration strategies (ε-greedy, softmax).
Analyze the impact of batch size and replay interval on performance.
3. Train the Agent:
Use the train() function to train the DQN agent in the Enduro environment.
Save models automatically after each training episode.
4. Evaluate and Visualize:
Use the provided plotting functions to visualize rewards, losses, and steps per episode.
Compare different hyperparameter configurations to identify the best-performing model.
5. Test the Trained Agent:
Load the saved model using the reconstruction cell.
Run the agent in the Enduro environment for real-time evaluation.