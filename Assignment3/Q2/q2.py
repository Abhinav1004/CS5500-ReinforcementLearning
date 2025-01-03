
import sys
import os
import gym 
import matplotlib.pyplot as plt 
import numpy as np

# from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 
import os
import gym
import numpy as np
import torch
import argparse

# Code of Policy Gradient Network (PGN).
# To implement neural layers.
# To implement activation functions.
# To optimize the weights.

class PGN(nn.Module):
    def __init__(self, learning_rate, input_size, actions_num, model_file):
        super(PGN, self).__init__() #5
        self.fcl1 = nn.Linear(*input_size, 128) #6 
        self.fcl2 = nn.Linear(128, 128)
        self.fcl3 = nn.Linear(128, actions_num)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #7 
        self.model_file = model_file
            
    def forward(self, state):
        x = F.relu(self.fcl1(state))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x        
    
    def save_model(self):
        # print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))



class PG_Agent():
    def __init__(self, learning_rate, gamma, input_size, actions_num, model_file, reward_to_go=False, advantage_normalization=False,baseline_type="time-dependent"):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.reward_memory = []
        self.action_memory = []
        self.PGN = PGN(learning_rate, input_size, actions_num, model_file)
        self.reward_to_go = reward_to_go
        self.advantage_normalization = advantage_normalization
        self.baseline_type = baseline_type
        
    def store_reward(self, reward):
        self.reward_memory.append(reward)
        
    def store_action(self, action):
        self.action_memory.append(action)
        
    def act(self, state):
        state = T.tensor([state], dtype=T.float).to(self.PGN.device)
        action_values = self.PGN.forward(state)
        action_probabilities = F.softmax(action_values, dim=-1)
        actions_chances = T.distributions.Categorical(action_probabilities)
        action = actions_chances.sample()
        action_log = actions_chances.log_prob(action)
        self.store_action(action_log)     
        
        return action.item()
        
    def learn(self):
        self.PGN.optimizer.zero_grad()
    
        # Compute returns (reward-to-go if enabled)
        G_t = np.zeros_like(self.reward_memory, dtype=np.float64)
        if self.reward_to_go:
            for t in range(len(self.reward_memory)):
                G_t_sum = 0
                discount = 1
                for k in range(t, len(self.reward_memory)):
                    G_t_sum += self.reward_memory[k] * discount
                    discount *= self.gamma
                G_t[t] = G_t_sum
        else:
            G_t_sum = sum([self.reward_memory[k] * (self.gamma ** k) for k in range(len(self.reward_memory))])
            G_t[:] = G_t_sum
    
        # Convert G_t to tensor and apply baseline if advantage normalization is enabled
        G_t = T.tensor(G_t, dtype=T.float).to(self.PGN.device)
    
        if self.advantage_normalization:
            if self.baseline_type == "time-dependent":
                # Time-dependent baseline: subtract the average return at each time step
                baseline = np.cumsum(self.reward_memory) / (np.arange(len(self.reward_memory)) + 1)
                baseline = T.tensor(baseline, dtype=T.float).to(self.PGN.device)
                G_t = G_t - baseline  # Subtract baseline
    
            # Normalize the advantage (after applying baseline)
            G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)
    
        # Compute policy gradient loss
        loss = 0
        for g_t_, log_probability in zip(G_t, self.action_memory):
            loss += -g_t_ * log_probability
    
        loss.backward()
        self.PGN.optimizer.step()
    
        # Clear memory
        self.action_memory = []
        self.reward_memory = []




# Plot results with reward-to-go and advantage normalization flags
def plot_learning_curve(env_name,reward_records, file, reward_to_go, advantage_normalization,batch_size):
    # Generate recent 100 interval average
    average_reward = []
    for idx in range(len(reward_records)):
        if idx < 100:
            avg_list = reward_records[:idx + 1]
        else:
            avg_list = reward_records[idx - 99:idx + 1]
        average_reward.append(np.mean(avg_list))

    # Determine plot title based on settings
    title = f"Learning Curve for Policy Gradient for {env_name}"
    subtitle = f"Reward-to-Go: {'True' if reward_to_go else 'False'}, " \
               f"Advantage Normalization: {'True' if advantage_normalization else 'False'},"\
               f"Batch size: {batch_size}"

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(reward_records, label='Total Rewards per Episode', alpha=0.5)
    plt.plot(average_reward, label='100-Episode Average Reward', color='orange')
    plt.title(f"{title}\n{subtitle}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()
    plt.savefig(file)
    plt.show()


def generate_results(env_name,input_dims,n_actions,learning_rate,num_games,batch_size,gamma,\
                     reward_to_go,advantage_normalization,baseline_type="time-dependent"):
    

    env = gym.make(env_name, render_mode="rgb_array")
    file_name = "PG_" + env_name + "_" + str(learning_rate) + "_" + str(num_games)+\
                "_rtg_"+str(reward_to_go)+"_adv_"+str(advantage_normalization)+"_batch_size_"+str(batch_size)
    
    scores_plot_file = os.path.join(current_dir, "plots", file_name + ".png")
    model_file = os.path.join(current_dir, "models", file_name)

    # Initialize agent with reward-to-go and advantage normalization options
    agent = PG_Agent(learning_rate, gamma=0.99, input_size=input_dims, actions_num=n_actions, model_file=model_file, 
                     reward_to_go=reward_to_go, advantage_normalization=advantage_normalization, baseline_type="time-dependent")
    
    mode = "train"  # Select among {"train", "test"}

    if mode == "train": 
        scores = []
        best_avg_score = -np.inf

        batch_rewards = []
        batch_actions = []
        
        for t in range(num_games):
            done = False
            score = 0
            state = env.reset()[0]
            step = 0
            episode_rewards = []
            while not done:
                step += 1
                action = agent.act(state)
                state_, reward, done, info, _ = env.step(action)
                score += reward 
                episode_rewards.append(reward)  # Store episode rewards
                agent.store_reward(reward)
                state = state_
                if step >= env_max_num_steps:
                    done = True
            
            # Store the rewards for the batch
            batch_rewards.append(episode_rewards)
            scores.append(score)
            
            # If batch size is reached, update the policy
            if (t + 1) % batch_size == 0:
                agent.learn()  # Learn after batch of episodes
                agent.reward_memory = []  # Clear rewards after learning

            avg_score = np.mean(scores[-100:])
            print("Episode", t, "- score %.2f" % score, "- avg_score %.2f" % avg_score)
            
            # Save the model if the average score improves
            if avg_score > best_avg_score:
                agent.PGN.save_model()
                best_avg_score = avg_score

        plot_learning_curve(env_name, scores, scores_plot_file, reward_to_go=reward_to_go,\
                            advantage_normalization=advantage_normalization,batch_size=batch_size)



# Parameters
gamma = 0.99
mode = "train"
env_name = "LunarLander-v2" # to prepare the results for Lunar Lander
reward_to_go=False
advantage_normalization=True
# env_name = "CartPole-v0"
baseline_type = "time-dependent"
batch_size = 5  # Number of episodes per batch


current_dir = "/Users/abhjha8/ml_projects/openai_gym/assignment_3/Q2/training_outputs"
current_dir = os.path.join(os.getcwd(),"training_outputs")
if not os.path.exists(os.path.join(current_dir,"models")):
    os.makedirs(os.path.join(current_dir,"models"))
if not os.path.exists(os.path.join(current_dir,"plots")):
    os.makedirs(os.path.join(current_dir,"plots"))

# if __name__ == "__main__":
    
#     env = gym.make(env_name, render_mode="rgb_array")
    
#     # Configure environment and agent parameters based on environment
#     if env_name == "LunarLander-v2":
#         env_max_num_steps = 1000
#         input_dims = [8]  # LunarLander state has 8 dimensions
#         n_actions = 4     # LunarLander has 4 discrete actions
#         num_games = 2000
#         learning_rate = 0.0005
#     elif env_name == "CartPole-v0":
#         env_max_num_steps = 200
#         input_dims = [4]  # CartPole state has 4 dimensions
#         n_actions = 2     # CartPole has 2 discrete actions (left or right)
#         num_games = 2000
#         learning_rate = 0.001
#     else:
#         raise ValueError("Unsupported environment")

#     generate_results(env_name,input_dims,n_actions,learning_rate,num_games,batch_size=5,gamma=0.99,\
#                      reward_to_go=reward_to_go,advantage_normalization=advantage_normalization,baseline_type="time-dependent")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Policy Gradient agent on OpenAI Gym environments")

    # Add command-line arguments with default values
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--mode', type=str, default="train", help='Mode: train or test (default: train)')
    parser.add_argument('--env_name', type=str, default="LunarLander-v2", help='Gym environment name (default: LunarLander-v2)')
    parser.add_argument('--reward_to_go', action='store_true', default=False, help='Use reward-to-go (default: False)')
    parser.add_argument('--advantage_normalization', action='store_true', default=True, help='Use advantage normalization (default: True)')
    parser.add_argument('--baseline_type', type=str, default="time-dependent", help='Baseline type (default: time-dependent)')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of episodes per batch (default: 5)')
    parser.add_argument('--current_dir', type=str, default=os.path.join(os.getcwd(),"training_outputs"), 
                        help='Directory for saving models and plots (default: current working directory)')

    args = parser.parse_args()

    # Ensure output directories exist
    current_dir = args.current_dir
    if not os.path.exists(os.path.join(current_dir, "models")):
        os.makedirs(os.path.join(current_dir, "models"))
    if not os.path.exists(os.path.join(current_dir, "plots")):
        os.makedirs(os.path.join(current_dir, "plots"))

    # Configure environment and agent parameters based on the selected environment
    if args.env_name == "LunarLander-v2":
        env_max_num_steps = 1000
        input_dims = [8]  # LunarLander state has 8 dimensions
        n_actions = 4     # LunarLander has 4 discrete actions
        num_games = 2000
        learning_rate = 0.0005
    elif args.env_name == "CartPole-v0":
        env_max_num_steps = 200
        input_dims = [4]  # CartPole state has 4 dimensions
        n_actions = 2     # CartPole has 2 discrete actions (left or right)
        num_games = 2000
        learning_rate = 0.001
    else:
        raise ValueError("Unsupported environment")

     # Call the function to generate results with command-line argument values
    generate_results(args.env_name, input_dims, n_actions, learning_rate, num_games, args.batch_size, args.gamma,
                     args.reward_to_go, args.advantage_normalization, args.baseline_type)
    
