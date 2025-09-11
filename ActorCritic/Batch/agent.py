import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Compute discounted rewards
def discount_rewards(rewards, gamma):
    discounted = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    return discounted

# Compute advantages
def compute_advantage(values, td_target = None, returns_to_go = None, mode = "baseline"):
    advantages = torch.zeros_like(values)

    if mode == "baseline":
        # Baseline - Not biased but high variance
        for i in range(len(values)):
            advantages[i] = returns_to_go[i] - values[i] 

    elif mode == "TD":
        # One-step TD - Biased but low variance
        with torch.no_grad():
            advantages = td_target - values
    return advantages


class Actor(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_dim=128, seed=None):
        super().__init__()
        
        # Set seed for consistent weight initialization
        if seed is not None:
            torch.manual_seed(seed)
            
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden_dim    
        self.tanh = torch.nn.Tanh()

        # Actor Network
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self): 
        # Weights initialization
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        # Actor forward pass
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        
        return normal_dist
    
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed=None):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = 128
        self.relu = torch.nn.ReLU()

        # Critic network 
        self.fc1_critic = torch.nn.Linear(state_dim, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()
        
        
    def init_weights(self): 
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.zeros_(m.bias)

    def forward(self, v):
        # critic forward pass
        v_critic = self.relu(self.fc1_critic(v))
        v_critic = self.relu(self.fc2_critic(v_critic))
        value = self.fc3_critic_value(v_critic)

        return value
        
        
        
class Agent(object):
    def __init__(self, actor, critic, gamma=0.99, device='cpu'):
        self.train_device = torch.device(device)
        self.actor = actor.to(self.train_device)
        self.critic = critic.to(self.train_device)

        # Optimizations for GPU speed (mixed precision)
        self.is_cuda = self.train_device.type == 'cuda'
        self.use_amp = self.is_cuda and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda') 

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.batch_size = 0 
        self.states_lst = []
        self.next_states_lst = []
        self.action_log_probs_lst = []
        self.rewards_lst = []
        self.done_lst = []

    def update_critic(self, values, returns):
        # critic update
        self.critic.train()
        self.critic_optimizer.zero_grad()
        loss = F.mse_loss(values, returns)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        self.critic.eval() 
        return

    def update_actor(self, action_log_probs, advantages):
        # actor update
        self.actor.train()
        self.actor_optimizer.zero_grad()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss = -(action_log_probs * advantages).mean()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        self.actor.eval() 
        return

    def update(self, mode="baseline"):
        
        # Prepare batch data
        batch_action_log_probs = []
        batch_advantages = []
        batch_returns = []
        batch_values = []
        batch_td_targets = []
        for eps in range(self.batch_size): 
            
            # Exctract data for episode eps
            action_log_probs = torch.stack(self.action_log_probs_lst[eps], dim=0)
            if self.is_cuda and action_log_probs.device != self.train_device:
                action_log_probs = action_log_probs.to(self.train_device, non_blocking=True)
            else:
                action_log_probs = action_log_probs.to(self.train_device)
            rewards = torch.stack(self.rewards_lst[eps], dim=0)
            if self.is_cuda and rewards.device != self.train_device:
                rewards = rewards.to(self.train_device, non_blocking=True)
            else:
                rewards = rewards.to(self.train_device)
            rewards = rewards.squeeze(-1)
            states = torch.stack(self.states_lst[eps], dim=0)
            if self.is_cuda and states.device != self.train_device:
                states = states.to(self.train_device, non_blocking=True)
            else:
                states = states.to(self.train_device)
            
            # compute returns to go, values and estimated values (bootstrap) 
            return_to_go = discount_rewards(rewards, self.gamma).to(self.train_device).squeeze(-1).detach()
            values = self.critic(states).squeeze(-1)
            td_target = rewards + self.gamma * torch.cat([values[1:], torch.tensor([0.0], device=self.train_device)])
            
            # compute advantages (different version) for actor update
            advantages = compute_advantage(values, td_target=td_target, 
                                           returns_to_go=return_to_go, mode=mode).detach()
            # store for batch update
            batch_action_log_probs.append(action_log_probs)
            batch_advantages.append(advantages)
            batch_returns.append(return_to_go)
            batch_values.append(values)
            batch_td_targets.append(td_target)
        # concatenation of all episodes in the batch
        batch_action_log_probs = torch.cat(batch_action_log_probs, dim=0)
        batch_advantages = torch.cat(batch_advantages, dim=0)
        batch_returns = torch.cat(batch_returns, dim=0)
        batch_values = torch.cat(batch_values, dim=0)
        batch_td_targets = torch.cat(batch_td_targets, dim=0)
        self.update_actor(batch_action_log_probs, batch_advantages)
        self.update_critic(batch_values, batch_td_targets)
        self.clear_storage()
        return

    def get_action(self, state, evaluation=False):
        # Given state, sample action from the policy
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.actor(x)
        if evaluation:  # Return mean
            return normal_dist.mean, None
        else:   # Sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, states, next_states, action_log_probs, rewards, dones):
        # Store the transition in the buffer
        if self.is_cuda:
            states = [torch.from_numpy(arr).float().to(self.train_device, non_blocking=True) for arr in states]
            next_states = [torch.from_numpy(arr).float().to(self.train_device, non_blocking=True) for arr in next_states]
            rewards = [torch.tensor(reward, device=self.train_device, dtype=torch.float32) for reward in rewards]
        else:
            states = [torch.from_numpy(arr).float() for arr in states]
            next_states = [torch.from_numpy(arr).float() for arr in next_states]
            rewards = [torch.Tensor([reward]) for reward in rewards]
        self.states_lst.append(states)
        self.next_states_lst.append(next_states)
        self.action_log_probs_lst.append(action_log_probs)
        self.rewards_lst.append(rewards)
        self.done_lst.append(dones)
        self.batch_size = self.batch_size + 1
        return

    def clear_storage(self):
        # Reset storage
        self.states_lst = []
        self.next_states_lst = []
        self.action_log_probs_lst = []
        self.rewards_lst = []
        self.done_lst = []
        self.batch_size = 0
        return



