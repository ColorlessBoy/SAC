import gym
import torch
import os
import csv
from time import time
import numpy as np

from models import ActorCritic
from utils import EnvSampler, EnvSampler2, hard_update
from pg2 import PG

def run(args):
    env = gym.make(args.env)

    device = torch.device(args.device)

    # 1. Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # env.seed(args.seed)

    # 2. Create nets. 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    policy_hidden_sizes = (256, 256, 256)
    q_hidden_sizes = (256, 256, 256)
    ac = ActorCritic(state_size, action_size, 
                    policy_hidden_sizes, q_hidden_sizes, 
                    activation=torch.nn.LeakyReLU(inplace=True),
                    generative=True).to(device)
    ac_target = ActorCritic(state_size, action_size, 
                    policy_hidden_sizes, q_hidden_sizes, 
                    activation=torch.nn.LeakyReLU(inplace=True),
                    generative=True).to(device)
    hard_update(ac, ac_target)

    # env_sampler = EnvSampler(env, max_episode_step=4000, capacity=1e6)
    env_sampler = EnvSampler2(env, gamma=args.gamma1, capacity=1e6)

    alg =   PG(ac, ac_target,
                gamma=args.gamma2, alpha=0.2,
                q_lr=1e-3, pi_lr=1e-3, target_lr = 5e-3,
                device=device)

    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return ac_target.get_action(state, sigma=args.noise_sigma, clip=args.noise_clip) 

    def get_mean_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return ac_target.get_action(state, deterministic=True)

    start_time = time()
    for _ in range(args.start_steps):
        env_sampler.addSample()
    print("Warmup uses {}s.".format(time() - start_time))

    for step in range(1, args.total_steps+1):
        env_sampler.addSample(get_action)

        if step % args.update_every == 0:
            for _ in range(args.update_every):
                batch = env_sampler.sample(args.batch_size)
                losses = alg.update(*batch)
        
        if step % args.test_every == 0:
            test_reward = env_sampler.test(get_mean_action, test_gamma=args.gamma2)
            yield (step, test_reward, *losses)
    
    torch.save(ac.state_dict(), './env_{}_ac_time{}.pth.tar'.format(args.env, time()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--env', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default cpu)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='start steps')
    parser.add_argument('--total_steps', type=int, default=100000, metavar='N',
                        help='total epochs')
    parser.add_argument('--update_every', type=int, default=50, metavar='N',
                        help='update steps')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size')
    parser.add_argument('--test_every', type=int, default=10000, metavar='N',
                        help='test_every')
    parser.add_argument('--gamma1', type=float, default=0.999, metavar='F',
                        help='gamma in sampler')
    parser.add_argument('--gamma2', type=float, default=0.99, metavar='F',
                        help='gamma in PG')
    parser.add_argument('--noise_sigma', type=float, default=0.2, metavar='F',
                        help='noise sigma')
    parser.add_argument('--noise_clip', type=float, default=0.5, metavar='F',
                        help='noise clip')
    args = parser.parse_args()

    # Test Dir
    test_logdir = "./logs/env_{}".format(args.env)
    test_file_name = 'test_env_{}_batch{}_seed{}_total_steps{}_time{}.csv'.format( 
                    args.env, args.batch_size, args.seed, args.total_steps, time())
    if not os.path.exists(test_logdir):
        os.makedirs(test_logdir)
    test_full_name = os.path.join(test_logdir, test_file_name)
    test_csvfile = open(test_full_name, 'w')
    test_writer = csv.writer(test_csvfile)
    test_writer.writerow(['step', 'reward'])

    start_time = time()

    for step, test_reward, q_loss, pi_loss, target_loss in run(args):
        test_writer.writerow([step, test_reward])
        print("Step {}: Reward = {:>8.6f}, q_loss = {:>8.6f}, pi_loss = {:>8.6f}, target_loss = {:>8.6f}".format(
            step, test_reward, q_loss, (1 - args.gamma2)*pi_loss, target_loss
        ))

    print("Total time: {}s.".format(time() - start_time))