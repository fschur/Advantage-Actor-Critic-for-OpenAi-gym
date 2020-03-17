# Advantage-Actor-Critic for OpenAI-gym environments
Implementation of Advantage-Actor-Critic with entropy regularization in Pytorch for OpenAI-gym environments.

## Advantage-Actor-Critic
The policy gradient in Adavantage-Actor-Crititc differes from the classical REINFORCE policy gradient by using a baseline to reduce variance. This baseline is an approximation of the state value function (Critic). Since the baseline is not dependent on the action this does not introduce bias.  
For more detailed information I would recommend reading this [articel](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f).

## Entropy regularization
In order to encourage exploration we add the entropy of the policy distribution to the loss. This forces the actor to consider as much actions as possible while still maximizing the reward.
