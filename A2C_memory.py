import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.rewards = []
        self.action_prob = []
        self.state_values = []
        self.entropy = []

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

        return torch.stack(self.action_prob), torch.stack(self.state_values), \
               disc_rewards.to(device), torch.stack(self.entropy)

    def update(self, reward, entropy, log_prob, state_value):
        self.entropy.append(entropy)
        self.action_prob.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)

    def reset(self):
        del self.rewards[:]
        del self.action_prob[:]
        del self.state_values[:]
        del self.entropy[:]
