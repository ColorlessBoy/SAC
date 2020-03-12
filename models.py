import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# We require that action is in [-1, 1]^n
class GaussianPolicyNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 max_log_std=2, min_log_std=-20):
        super().__init__()
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

        self.net = mlp([input_size]+list(hidden_sizes), activation, activation)

        self.mu_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.std_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, state, deterministic=False, with_logprob=True):
        state_feature = self.net(state)
        mu = self.mu_layer(state_feature)
        log_std = self.std_layer(state_feature)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        pi_distribution = Normal(loc=mu, scale=log_std.exp())

        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        
        return pi_action, logp_pi

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_sizes=(256, 256), \
                activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.q = mlp([state_size + action_size] + list(hidden_sizes) + [1], activation, output_activation)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return q.squeeze(-1)

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_sizes=(256,256),
                 activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        self.pi = GaussianPolicyNetwork(state_size, action_size, hidden_sizes, activation)
        self.q1 = QNetwork(state_size, action_size, hidden_sizes, activation, output_activation)
        self.q2 = QNetwork(state_size, action_size, hidden_sizes, activation, output_activation)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.pi(state, deterministic, False)
        return action.cpu().numpy()[0]

if __name__ == '__main__':
    state_size = 10
    action_size = 4
    ac = ActorCritic(state_size, action_size)
    state = torch.randn((2, 10))
    action, _ = ac.pi(state)

    print("action = ", action)

    print(ac.q1(state, action))

    print(ac.q2(state, action))

    print(ac)