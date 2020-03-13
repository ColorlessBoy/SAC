from collections import namedtuple
import random
# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'done'))

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Memory(object):
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self.position = 0
        self.memory = []

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

class EnvSampler(object):
    def __init__(self, env, gamma=1, max_episode_step=1000, capacity=1e5):
        self.env = env
        self.gamma = gamma
        self.max_episode_step = max_episode_step
        self.action_scale = (env.action_space.high - env.action_space.low)/2
        self.action_bias = (env.action_space.high + env.action_space.low)/2
        self.memory = Memory(capacity)

        self.env_init()
    
    def env_init(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_step = 1

    # action_encode and action_decode project action into [-1, 1]^n
    def _action_encode(self, action):
        return (action - self.action_bias)/self.action_scale
    
    def _action_decode(self, action_):
        return action_ * self.action_scale + self.action_bias

    def addSample(self, get_action=None):
        if get_action:
            action_ = get_action(self.state)
        else:
            action_ = self.env.action_space.sample()
        action =self._action_decode(action_)

        state = self.state

        if self.gamma < 1 and self.gamma > 0:
            if random.random() < self.gamma:
                self.state, reward, self.done, _ = self.env.step(action) 
            else:
                self.env_init()
                reward, self.done = 0, False
        else:
            self.state, reward, self.done, _ = self.env.step(action)
            self.episode_step += 1
            if self.episode_step >= self.max_episode_step:
                self.done = True

        self.memory.push(state, action_, reward, self.state, self.done)

        if self.done:
            self.env_init()
    
    def sample(self, batch_size):
        return self.memory.sample(batch_size)
    
    def test(self, get_action, times=10):
        episode_reward = 0.0
        episode_step = 1
        for _ in range(times):
            self.env_init()
            while(not self.done):
                action_ = get_action(self.state)
                action =self._action_decode(action_)
                next_state, reward, self.done, _ = self.env.step(action) 
                self.state = next_state
                episode_reward += reward
                episode_step += 1
                if episode_step >= self.max_episode_step:
                    self.done = True
        self.env_init()
        return episode_reward / times