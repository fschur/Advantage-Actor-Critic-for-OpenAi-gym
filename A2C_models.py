import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCriticContinuous(nn.Module):
    """
    Actor-Critic for continuous action spaces. The network returns a state_value (critic) and
    action mean and action standarddeviation (actor). The action is the sampled from a normal
    distribution with mean and std given by the actor.
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticContinuous, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head
        self.critic_head = nn.Linear(int(hidden_dim/2), 1)

        # actor head
        self.actor_head_mean = nn.Linear(int(hidden_dim/2), action_dim)
        self.actor_head_sigma = nn.Linear(int(hidden_dim / 2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))

        # how good is the current state?
        state_value = self.critic_head(x)

        action_mean = (self.actor_head_mean(x))
        action_sigma = F.softplus(self.actor_head_sigma(x) + 0.0001)

        return action_mean, action_sigma, state_value


class ActorCriticDiscrete(nn.Module):
    """
    Actor-Critic for discrete action spaces. The network returns a state_value (critic)and action probabilities (actor).
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticDiscrete, self).__init__()

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

