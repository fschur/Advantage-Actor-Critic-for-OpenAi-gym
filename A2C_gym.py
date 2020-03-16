import gym
import torch
import numpy as np
from A2C_models import ActorCriticContinuous, ActorCriticDiscrete
from A2C_memory import Memory
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR


"""
Implementation of Advantage-Actor-Critic for gym environments
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_action(model, state, mode):
    state = torch.Tensor(state).to(device)
    if mode == "continuous":
        mean, sigma, state_value = model(state)
        s = torch.distributions.MultivariateNormal(mean, torch.diag(sigma))
    else:
        probs, state_value = model(state)
        s = Categorical(probs)

    action = s.sample()
    entropy = s.entropy()

    return action.numpy(), entropy, s.log_prob(action), state_value


def evaluate(actor_critic, env, repeats, mode):
    actor_critic.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                if mode == "continuous":
                    mean, sigma, _ = actor_critic(state)
                    m = torch.distributions.Normal(mean, sigma)
                else:
                    probs, _ = actor_critic(state)
                    m = Categorical(probs)

            action = m.sample()
            state, reward, done, _ = env.step(action.numpy())
            perform += reward
    actor_critic.train()
    return perform/repeats


def train(memory, optimizer, gamma, eps):
    action_prob, values, disc_rewards, entropy = memory.calculate_data(gamma)

    advantage = disc_rewards.detach() - values

    policy_loss = (-action_prob*advantage.detach()).mean()
    value_loss = 0.5 * advantage.pow(2).mean()
    loss = policy_loss + value_loss - eps*entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main(gamma=0.99, lr=1e-2, num_episodes=400, eps=0.001, seed=42, lr_step=100, lr_gamma=0.95, measure_step=100, 
         measure_repeats=100, horizon=np.inf, hidden_dim=64, env_name='CartPole-v1', render=True):
    """
    :param gamma: reward discount factor
    :param lr: initial learning rate
    :param num_episodes: total number of episodes performed in the environment
    :param eps: entropy regularization parameter (increases exploration)
    :param seed: random seed
    :param lr_step: every "lr_step" many episodes the lr is updated by the factor "lr_gamma"
    :param lr_gamma: see above
    :param measure_step: every "measure_step" many episodes the the performance is measured using "measure_repeats" many
    episodes
    :param measure_repeats: see above
    :param horizon: if not set to infinity limits the length of the episodes when training
    :param hidden_dim: hidden dimension used for the DNN
    :param env_name: name of the gym environment
    :param render: if True the environment is rendered twice every "measure_step" many episodes
    """
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)

    # check whether the environment has a continuous or discrete action space.
    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    # Get number of actions for the discrete case and action dimension for the continuous case.
    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    if action_mode == "continuous":
        actor_critic = ActorCriticContinuous(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    else:
        actor_critic = ActorCriticDiscrete(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    performance = []
    for episode in range(num_episodes):
        # reset memory
        memory = Memory()
        # display the episode_performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(actor_critic, env, measure_repeats, action_mode)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr())

        state = env.reset()

        done = False
        count = 0
        while not done and count < horizon:
            count += 1
            action, entropy, log_prob, state_value = select_action(actor_critic, state, action_mode)
            state, reward, done, _ = env.step(action)
            if render and episode % int((measure_step/2)) == 0:
                env.render()

            # save the information
            memory.update(reward, entropy, log_prob, state_value)

        # train on the observed data
        train(memory, optimizer, gamma, eps)
        # update the learning rate
        scheduler.step()

    return actor_critic, performance


if __name__ == '__main__':
    main()
