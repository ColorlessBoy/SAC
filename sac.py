import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

class SAC(object):
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
        self.pi_optim = Adam(self.ac.pi.parameters(), lr = pi_lr)
        self.q1_optim = Adam(self.ac.q1.parameters(), lr = q_lr) 
        self.q2_optim = Adam(self.ac.q2.parameters(), lr = q_lr)
        self.target_optim = SGD(self.ac_target.parameters(),  lr = target_lr)

    def update(self, state, action, reward, nstate, done):
        state  = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        nstate = torch.FloatTensor(nstate).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q1_loss, q2_loss = self.update_q_net(state, action, reward, nstate, done)
        pi_loss = self.update_pi_net(state)
        target_loss = self.update_target_net(state)

        return q1_loss, q2_loss, pi_loss, target_loss
    
    def update_q_net(self, state, action, reward, nstate, done):
        with torch.no_grad():
            naction, log_pi_naction = self.ac_target.pi(nstate)
            q1_target = self.ac_target.q1(nstate, naction)
            q2_target = self.ac_target.q2(nstate, naction)
            q_target = torch.min(q1_target, q2_target)
            backup = reward + self.gamma * (1 - done) * (q_target - self.alpha * log_pi_naction)

        q1 = self.ac.q1(state, action)
        q2 = self.ac.q2(state, action)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        return q1_loss, q2_loss
    
    def update_pi_net(self, state):
        action, log_pi_action = self.ac.pi(state)

        q = torch.min(self.ac.q1(state, action), self.ac.q2(state, action))

        pi_loss = torch.mean(self.alpha * log_pi_action - q)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        return pi_loss
    
    def update_target_net(self, state):
        target_loss = 0.0
        for param, target_param in zip(self.ac.parameters(), self.ac_target.parameters()):
            target_loss += 0.5 * torch.sum((param - target_param)**2)

        self.target_optim.zero_grad()
        target_loss.backward()
        self.target_optim.step()

        return target_loss