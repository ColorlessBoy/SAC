import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
import itertools

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
    
    def get_q_loss(self, state, action, reward, nstate, done, ap=0.0):
        with torch.no_grad():
            naction, log_pi_naction = self.ac_target.pi(nstate)
            q1_target = self.ac_target.q1(nstate, naction)
            q2_target = self.ac_target.q2(nstate, naction)
            q_target = torch.min(q1_target, q2_target)
            backup = reward + self.gamma * (1 - done) * q_target

        action.requires_grad = True
        q1 = self.ac.q1(state, action)
        q2 = self.ac.q2(state, action)

        action_penalty = 0.0
        if ap > 0.0:
            gradient = torch.autograd.grad((q1+q2).mean(), action, retain_graph=True)[0]
            action_penalty = ap * gradient.pow(2).mean()

        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup) + action_penalty

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