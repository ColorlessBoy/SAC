import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
import itertools
import numpy as np

class PG(object):
    def __init__(self, ac, ac_target,
                gamma=0.99, alpha=0.2,
                q_lr=1e-3, pi_lr=1e-3, target_lr = 5e-3,
                device=torch.device('cpu')):
        # nets
        self.ac = ac
        self.ac_target = ac_target

        # hyperparameters
        self.gamma, self.alpha = gamma, alpha

        # device
        self.device = device

        # optimization
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_optim = Adam(self.ac.pi.parameters(), lr = pi_lr)
        self.q_optim  = Adam(self.q_params, lr = q_lr)
        self.target_optim = SGD(self.ac_target.parameters(),  lr = target_lr)

    def update(self, state, action, reward, nstate, done):
        state  = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        nstate = torch.FloatTensor(nstate).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        self.q_optim.zero_grad()
        q_loss = self.get_q_loss(state, action, reward, nstate, done)
        q_loss.backward()
        self.q_optim.step()

        self.pi_optim.zero_grad()
        pi_loss = self.get_pi_loss(state)
        pi_loss.backward()
        self.pi_optim.step()

        self.target_optim.zero_grad()
        target_loss = self.get_target_loss(state)
        target_loss.backward()
        self.target_optim.step()

        return q_loss, pi_loss, target_loss
    
    def get_q_loss(self, state, action, reward, nstate, done):
        beta = 0.05

        with torch.no_grad():
            # Bellman Equation
            naction, _ = self.ac_target.pi(nstate)
            q1_target = self.ac_target.q1(nstate, naction)
            q2_target = self.ac_target.q2(nstate, naction)
            q_target = torch.min(q1_target, q2_target)
            backup = reward + self.gamma * (1 - done) * q_target

            backup2 = backup
            if beta > 0.0:
                # Entropy
                expand_batch = 1000
                ns = nstate.repeat(expand_batch, 1)
                na = 2 * torch.rand(ns.shape[0], naction.shape[-1], device=self.device) - 1
                nq = torch.min(self.ac_target.q1(ns, na), self.ac_target.q2(ns, na))
                q_target2 = self.alpha * (naction.shape[-1] * np.log(2) 
                            + (nq / self.alpha).exp().reshape(expand_batch, -1).mean(dim=0).log().clamp(-20, 20))
                backup2 = reward + self.gamma * (1 - done) * q_target2

        q1 = self.ac.q1(state, action)
        q2 = self.ac.q2(state, action)

        q_loss = (1 - beta) * F.mse_loss(q1, backup) + (1 - beta) * F.mse_loss(q2, backup) \
            + beta * F.mse_loss(q1, backup2) + beta * F.mse_loss(q2, backup2)
        return q_loss
    
    def get_pi_loss(self, state):
        action, log_pi_action = self.ac.pi(state)
        q = torch.min(self.ac.q1(state, action), self.ac.q2(state, action))
        pi_loss = torch.mean(-q)
        return pi_loss
    
    def get_target_loss(self, state):
        target_loss = 0.0
        for param, target_param in zip(self.ac.parameters(), self.ac_target.parameters()):
            target_loss += 0.5 * torch.sum((param - target_param)**2)
        return target_loss