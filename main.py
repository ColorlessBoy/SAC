import gym
import torch
import os
import csv
from time import time
import random


from models import ActorCritic
from utils import EnvSampler, hard_update
from sac import SAC

def run(args):
    env = gym.make(args.env_name)

    device = torch.device(args.device)

    # 1. Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    # 2. Create nets. 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    hidden_sizes = (256, 256)
    ac = ActorCritic(state_size, action_size, hidden_sizes).to(device)
    ac_target = ActorCritic(state_size, action_size, hidden_sizes).to(device)
    hard_update(ac, ac_target)

    env_sampler = EnvSampler(env, max_episode_step=1000, capacity=1e5)

    alg =   SAC(ac, ac_target,
                gamma=0.99, alpha=0.2,
                q_lr=1e-3, pi_lr=1e-3, target_lr = 5e-3,
                device=device)

    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return ac_target.get_action(state)

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
            test_reward = env_sampler.test(get_mean_action, 10)
            yield (step, test_reward, *losses)
    
    torch.save(ac.pi.state_dict(), './env_{}_pi_net.pth.tar'.format(args.env_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--env_name', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default cpu)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='start steps (default: 0)')
    parser.add_argument('--total_steps', type=int, default=100000, metavar='N',
                        help='total epochs (default: 0)')
    parser.add_argument('--update_every', type=int, default=100, metavar='N',
                        help='update steps (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size (default: 0)')
    parser.add_argument('--test_every', type=int, default=10000, metavar='N',
                        help='test_every (default: 0)')
    args = parser.parse_args()

    # Test Dir
    test_logdir = "./logs/env_{}".format(args.env_name)
    test_file_name = 'test_env_{}_batch{}_seed{}_total_steps{}_time{}.csv'.format( 
                    args.env_name, args.batch_size, args.seed, args.total_steps, time())
    if not os.path.exists(test_logdir):
        os.makedirs(test_logdir)
    test_full_name = os.path.join(test_logdir, test_file_name)
    test_csvfile = open(test_full_name, 'w')
    test_writer = csv.writer(test_csvfile)
    test_writer.writerow(['step', 'reward'])

    start_time = time()

    for step, test_reward, q1_loss, q2_loss, pi_loss, target_loss in run(args):
        test_writer.writerow([step, test_reward])
        print("Step {}: Reward = {:>10.6f}, q1_loss = {:>8.6f}, q2_loss = {:>8.6f}, pi_loss = {:>8.6f}, target_loss = {:>8.6f}".format(
            step, test_reward, q1_loss, q2_loss, pi_loss, target_loss
        ))

    print("Total time: {}s.".format(time() - start_time))