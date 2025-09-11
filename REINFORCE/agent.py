import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# to-go reward 
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t] # add next step 
        discounted_r[t] = running_add 
    return discounted_r

class Policy(torch.nn.Module):
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


class Agent(object):
    def __init__(self, policy, gamma=0.99, device='cpu'):
        self.train_device = torch.device(device)
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3) 

        self.gamma = gamma
    
        self.is_cuda = self.train_device.type == 'cuda'
        self.use_amp = self.is_cuda and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        self.batch_size = 0 
        self.states_lst = []
        self.next_states_lst = []
        self.action_log_probs_lst = []
        self.rewards_lst = []
        self.done_lst = []


    def update_actor(self, action_log_probs, advantages):
        # actor update
        self.policy.train()
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                # Normalize advantages for stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                loss = -(action_log_probs * advantages).mean()
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        else:
            self.optimizer.zero_grad()
            
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss = -(action_log_probs * advantages).mean()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        self.policy.eval() 
        return

    def update_policy(self, baseline = 0.0, normalization=True):
        batch_action_log_probs = []
        batch_return = []

        for e in range(self.batch_size): 
            action_log_probs = torch.stack(self.action_log_probs_lst[e], dim=0)
            if self.is_cuda and action_log_probs.device != self.train_device:
                action_log_probs = action_log_probs.to(self.train_device, non_blocking=True)
            elif not self.is_cuda:
                action_log_probs = action_log_probs.to(self.train_device)
            
            rewards = torch.stack(self.rewards_lst[e], dim=0)
            if self.is_cuda and rewards.device != self.train_device:
                rewards = rewards.to(self.train_device, non_blocking=True)
            elif not self.is_cuda:
                rewards = rewards.to(self.train_device)
            rewards = rewards.squeeze(-1)

            returns_to_go = discount_rewards(rewards, self.gamma).squeeze(-1)
            
            batch_action_log_probs.append(action_log_probs)
            batch_return.append(returns_to_go)
            
        batch_action_log_probs = torch.cat(batch_action_log_probs, dim=0)
        batch_advantages = torch.cat(batch_return, dim=0)

        self.update_actor(batch_action_log_probs, batch_advantages - baseline)
        
        # Clear storage
        self.clear_storage()
        
        return        


    def get_action(self, state, evaluation=False):
        # Given state, sample action from the policy
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            mean = normal_dist.mean
            return mean.detach(), None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum(dim = -1)

            return action, action_log_prob


    def store_outcome(self, states, next_states, action_log_probs, rewards, dones):
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
        
        # update batch size
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
