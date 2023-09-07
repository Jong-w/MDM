from logger import Logger
from MDM import HONET, mp_loss
from utils import make_envs, take_action, init_obj
from storage import Storage
import wandb
import pickle
import gzip
import argparse
import torch
import cv2
import numpy as np
parser = argparse.ArgumentParser(description='Honet')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--dynamic', type=int, default=0,
                    help='dynamic_neural_network or not')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='FrostbiteNoFrameskip-v4',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=400,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e5),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')


# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--gamma-5', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-4', type=float, default=0.99,
                    help="discount factor supervisor")
parser.add_argument('--gamma-3', type=float, default=0.99,
                    help="discount factor manager")
parser.add_argument('--gamma-2', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-1', type=float, default=0.95,
                    help="discount factor supervisor")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=float(1e-7),
                    help='Random Gausian goal for exploration')
parser.add_argument('--hidden-dim-Hierarchies', type=int, default=[16, 256, 256, 256, 256],
                    help='Hidden dim (d)')
parser.add_argument('--time_horizon_Hierarchies', type=int, default=[1, 10, 15, 20, 25],
                    help=' horizon (c_s)')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='MDM',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

def experiment(args):

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, 'MDM_64', args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers)
    HONETS = HONET(
        num_workers=args.num_workers,
        input_dim=envs.observation_space.shape,
        hidden_dim_Hierarchies = args.hidden_dim_Hierarchies,
        time_horizon_Hierarchies=args.time_horizon_Hierarchies,
        n_actions=envs.single_action_space.n,
        dynamic=0,
        device=device,
        args=args)
    HONETS.load_state_dict(torch.load('models/FrostbiteNoFrameskip-v4_MDM_steps=100006400.pt', map_location=torch.device(device))['model'])
    HONETS.eval()

    step = 0
    goals_5, states_total, goals_4, goals_3, goals_2, masks = HONETS.init_obj()
    x = envs.reset()
    train_eps = float(1e-7)
    info_list = []
    hierarchy5_list = []
    hierarchy4_list = []
    hierarchy3_list = []
    steps_list = []

    reward_list = []

    while step < args.max_steps:
        # Detaching LSTMs and goals_m
        HONETS.repackage_hidden()
        goals_5 = [g.detach() for g in goals_5]
        goals_4 = [g.detach() for g in goals_4]
        goals_3 = [g.detach() for g in goals_3]
        goals_2 = [g.detach() for g in goals_2]

        with torch.no_grad():
            for _ in range(args.num_steps):
                action_dist, goals_5, states_total, value_5, goals_4, value_4, goals_3, value_3, goals_2, value_2, value_1, hierarchies_selected, train_eps \
                    = HONETS(x, goals_5, states_total, goals_4, goals_3, goals_2, masks[-1], step, train_eps)

                # Take a step, log the info, get the next state
                action, logp, entropy = take_action(action_dist)
                x, reward, done, info = envs.step(action)
                reward_list.append(reward[0])
                logger.log_episode(info, step)

                mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
                masks.pop(0)
                masks.append(mask)



                log_episode = {}
                for episode_dict in info:
                   if not 'final_info' in episode_dict:
                       if episode_dict['returns/episodic_reward'] is not None:
                           log_episode = ({"training/episode/reward": episode_dict['returns/episodic_reward'],
                                      "training/episode/length": episode_dict['returns/episodic_length'], 'step': step})
                   else:
                       if episode_dict['final_info'] is not None:
                           log_episode = ({"training/episode/reward": episode_dict['returns/episodic_reward'],
                                      "training/episode/length": episode_dict['returns/episodic_length'], 'step': step})

                info_list.append(info[0]['returns/episodic_reward'])
                hierarchy5_list.append(hierarchies_selected[:, 0].item())
                hierarchy4_list.append(hierarchies_selected[:, 1].item())
                hierarchy3_list.append(hierarchies_selected[:, 2].item())
                steps_list.append(step)
                step += args.num_workers

                # print if it is done
                # print(done[0])
                # print('reward:', reward[0])
                if done[0]:
                    # cv2.imwrite('curr{}.png'.format(step), (x * 255).squeeze())
                    x, done[0] = envs.reset(), False
                    goals_5, states_total, goals_4, goals_3, goals_2, masks = HONETS.init_obj()
                    print('*' * 10)
                    print('sum_reward:', np.sum(reward_list))
                    print('max_reward:', np.max(reward_list))
                    if np.max(reward_list)>100:
                        print('')
                    reward_list = []


    data = {"reward": info_list,
            "hierarchy5": hierarchy5_list,
            "hierarchy4": hierarchy4_list,
            "hierarchy3": hierarchy3_list}

    with gzip.open('testPickleFile.pickle', 'wb') as f:
        pickle.dump(data, f)

    envs.close()


def main(args):
    run_name = args.run_name
    for seed in range(1):
        # wandb.init(project="MDM",
        #            config=args.__dict__
        #            )
        # args.seed = seed
        # wandb.run.name = f"{run_name}_runseed={seed}"
        experiment(args)
        # wandb.finish()


if __name__ == '__main__':
    main(args)
