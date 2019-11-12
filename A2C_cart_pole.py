import gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

"""
Implementation of A2C for gym's CartPole.
"""


# Fully connected NN with Actor and Critic head
class ActorCritic(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head
        self.critic_head = nn.Linear(int(hidden_dim/2), 1)

        # actor head
        self.actor_head = nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))

        # how good is the current state?
        state_value = self.critic_head(x)

        # actor's probability to take each action
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        return action_prob, state_value


# memory to save the state, action, reward sequence from the current episode
class Memory:
    def __init__(self):
        self.rewards = []
        self.action_prob = []
        self.state_values = []
        self.entropy = 0

    def calculate_data(self, gamma):
        # compute the discounted rewards
        disc_rewards = []
        R = 0
        for reward in self.rewards[::-1]:
            R = reward + gamma*R
            disc_rewards.insert(0, R)

        # transform to tensor and normalize
        disc_rewards = torch.Tensor(disc_rewards)
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 0.001)

        return torch.stack(self.action_prob), torch.stack(self.state_values), disc_rewards, self.entropy

    def reset(self):
        del self.rewards[:]
        del self.action_prob[:]
        del self.state_values[:]
        self.entropy = 0


def select_action(model, state, memory):
    state = torch.Tensor(state)
    probs, state_value = model(state)
    entropy = -(probs*probs.log()).sum()
    # sample from the probability distribution given by the actor
    m = Categorical(probs)
    action = m.sample()

    # save the log-probabilities and the state_value
    memory.entropy += entropy
    memory.action_prob.append(m.log_prob(action))
    memory.state_values.append(state_value)

    return action.item()


def train(memory, optimizer, gamma):
    probs, values, disc_rewards, entropy = memory.calculate_data(gamma)

    advantage = disc_rewards - values

    policy_loss = (-probs*advantage).mean()
    value_loss = 0.5 * advantage.pow(2).mean()
    loss = policy_loss + value_loss + 0.001*entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    memory.reset()


def main(gamma=0.97, lr=3e-4, num_episodes=3000, render=False):
    env = gym.make('CartPole-v0')

    actor_critic = ActorCritic(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                               hidden_dim=256)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    memory = Memory()

    perform = 0
    for episode in range(num_episodes):
        # display the performance every 100 episodes
        if episode % 100 == 0:
            print("Episode: ", episode)
            print("rewards: ", perform/100)
            perform = 0

        state = env.reset()
        if render:
            env.render()

        done = False
        while not done:
            action = select_action(actor_critic, state, memory)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            memory.rewards.append(reward)
        perform += np.array(memory.rewards).sum()
        train(memory, optimizer, gamma)

    return actor_critic


if __name__ == '__main__':
    main()
