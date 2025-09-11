import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from actor import Actor, Critic
from collections import deque
from hparams import HyperParams as hp
from env.custom_hopper import *
from sklearn.preprocessing import StandardScaler
from vanilla import train_model, get_action

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


if __name__=="__main__":
    env = gym.make('CustomHopper-source-v0')
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    scaler = StandardScaler()
    scaler_fitted = False

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        scaler.mean_ = ckpt['scaler_mean']
        scaler.scale_ = ckpt['scaler_scale']
        scaler.var_ = ckpt['scaler_var']
        scaler.n_samples_seen_ = ckpt['scaler_n_samples']
        scaler_fitted = True

        print("Loaded OK ex. Scaler samples {}".format(scaler.n_samples_seen_))

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    episodes = 0
    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []
        states_to_fit = []
        while steps < 2048:
            episodes += 1
            state = env.reset()
            states_to_fit.append(state)
            score = 0
            for _ in range(10000):
                if args.render:
                    env.render()

                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                states_to_fit.append(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break
            scores.append(score)

        # Fit scaler solo la prima volta
        if not scaler_fitted:
            scaler.fit(np.array(states_to_fit))
            scaler_fitted = True
            
        # Normalizza gli stati in memoria
        for i in range(len(memory)):
            memory[i][0] = scaler.transform([memory[i][0]])[0]

        # Converti la deque/lista di memory in un numpy array con dtype=object
        # in modo da evitare ValueError quando vanilla.train_model esegue np.array(memory)
        memory = np.array(list(memory), dtype=object)

        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))


        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

        if iter % 100:
            score_avg = int(score_avg)
            print(f"Iterazione: {iter}, Episodio corrente: {episodes}, Step: {steps}, Ritorno medio: {score_avg}")
