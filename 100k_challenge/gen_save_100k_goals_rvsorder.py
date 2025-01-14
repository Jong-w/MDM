from logger import Logger
import gym
from gym.wrappers import AtariPreprocessing
from MDM_no_hd import MDM_no_hd
from MDM import MDM
from a3c import a3c
from feudalnet import FeudalNetwork
from utils import make_envs, take_action, init_obj, basic_wrapper, atari_wrapper
from storage import Storage
import wandb
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import torch
import gc
import numpy as np
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MDM')

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
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
parser.add_argument('--mlp', type=int, default=0,
                    help='toggle to feedforward ML architecture')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=10,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.95,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.99,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=int(1e-5),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

# simple rl model for knowledge transfer
class rl_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, device, mlp=False):
        super(rl_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions)
        ).to(device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)
        return x


class mlp_env(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # fc layers and deconv layers to revert back image that shape is 3, 84, 84
        # fc layers
        self.fc1 = torch.nn.Linear(256, 256)
        self.fc2 = torch.nn.Linear(256, 512 * 7 * 7)
        # deconv layers from hidden activation that has shape of batch, 256 to batch, 3, 84, 84
        self.deconv_layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 3, kernel_size=5, stride=3, padding=1),    # 28x28 -> 84x84
            torch.nn.Sigmoid()        )

    def forward(self, x ):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 512, 7, 7)
        x = self.deconv_layers(x)

        return x 
    

def experiment(args):

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, 'MDM_64', args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers)
    
    env_flat = gym.make(args.env_name)
    # env_flat = AtariPreprocessing(env_flat, grayscale_obs=False, scale_obs=True)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env_flat.unwrapped, gym.envs.atari.AtariEnv)
    if is_atari:
        wrapper_fn = atari_wrapper
    else:
        wrapper_fn = basic_wrapper
        

    if args.model_name == 'FuN':
        model = FeudalNetwork(
            num_workers=args.num_workers,
            input_dim=envs.observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)
    if args.model_name == 'a3c':
        model = a3c(
            num_workers=args.num_workers,
            input_dim=envs.observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)

    if args.model_name == 'FuN':
        path = '100k_challenge/models_new_testing_fun/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
        # path = 'models_new_testing_fun/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
    if args.model_name == 'a3c':
        path = '100k_challenge/models/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
        # path = 'models/' + args.env_name + "_" + args.model_name + "_steps=102400.pt"
    model.load_state_dict(torch.load(path)['model'])
    model.eval()

    # load rnn rl model  
    rl = rl_model(input_dim=256, hidden_dim=256, n_actions=envs.single_action_space.n, device=device, mlp=args.mlp).to(device)
    rl.train()
    mlp_deconv = mlp_env().to(device)
    mlp_deconv.train()

    optimizer = torch.optim.Adam(rl.parameters(), lr=1e-3)
    optimizer_deconv = torch.optim.Adam(mlp_deconv.parameters(), lr=1e-3)

    # In orther to avoid gradient exploding, we apply gradient clipping.
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)

    goals, states, masks = model.init_obj()
    
    x = envs.reset()
    _x = env_flat.reset()
    step = 0
    step_t_ep = 0
    break_flag = False
    while step < args.max_steps:

        # Detaching LSTMs and goals
        model.repackage_hidden()
        goals = [g.detach() for g in goals]
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                'adv_m', 'adv_w'])

        xs = [np.zeros_like(x)]*len(goals)
        xs.append(x)
        for _ in range(args.num_steps):
            action_dist, goals, states, value_m, value_w \
                = model(x, goals, states, masks[-1])

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)
            _x, _reward, _done, _info = env_flat.step(action[0])

            #            infos =[ ]
            #           for i in range(len(done)):
            #                # empty dict
            #                info_temp = {}

            #                for dict_name, dict_array in info.items():
            # add dict_name and dict_array[i] to info_temp
            #                    info_temp[dict_name] = dict_array[i]

            #                infos.append(info_temp)

            # logger.log_episode(info, step)

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)

            xs.pop(0)
            xs.append(x)

            packed = model.finding_goal_alike(x, states, goals, masks, xs, env_flat)

            if packed is not None:
                state_goal, losses, x_trues = packed
                print(f"losses: {np.array(losses).min()}")
                #where isthe argmin(losses)
                for iii in range(len(state_goal)):
                    # train the model for generating goals
                    _state_goal = state_goal[iii]
                    _x_true = x_trues[iii]

                    _x_true_valid = _x_true[model.c:] # current states
                    _x_true_in = _x_true[:-model.c] # states from model.c step before

                    # MAKE evertthing to torch.tensor
                    _x_true_valid = [torch.tensor(xt).to(device) for xt in _x_true_valid]
                    _x_true_in = [torch.tensor(xt).to(device) for xt in _x_true_in]

                    _x_true_valid = torch.stack(_x_true_valid).squeeze().to(device)
                    _x_true_in = torch.stack(_x_true_in).squeeze().to(device)

                    _state_goal_valid = _state_goal[:-model.c] # goals from model.c step before
                    _goals_valid = [sg[0] for sg in _state_goal_valid] 
                    _goals_valid = torch.stack(_goals_valid).squeeze().to(device)

                    for inepoch in range(100):

                        # _x_pred = mlp_deconv(_goals_valid.detach(), _x_true_in.detach())
                        _x_pred = mlp_deconv(_goals_valid.detach())
                        
                        # loss
                        loss = mse_loss(_x_pred, _x_true_valid.detach())

                        optimizer_deconv.zero_grad()
                        with torch.autograd.set_detect_anomaly(True):
                            loss.backward()
                        optimizer_deconv.step()

                        print(f"loss: {loss}")

                # make dirs with the name of the model and env
                dirname = 'gen_goal/' + args.model_name + "_" + args.env_name
                os.makedirs(dirname, exist_ok=True)

                # save model but including the model name
                torch.save(mlp_deconv.state_dict(), dirname + '/mlp_deconv_' + args.env_name + "_" + args.model_name + "_steps=102400.pt")
                # make figures comparing x_true and x_pred
                for iii in range(len(state_goal)):
                    print('iii:',iii)
                    os.makedirs(os.path.join(dirname, f'epoch_{iii}'), exist_ok=True)

                    _state_goal = state_goal[iii]
                    _x_true = x_trues[iii]

                    _x_true_valid = _x_true[model.c:]
                    _x_true_in = _x_true[:-model.c]

                    _x_true_valid = [torch.tensor(xt).to(device) for xt in _x_true_valid]
                    _x_true_in = [torch.tensor(xt).to(device) for xt in _x_true_in]

                    _x_true_valid = torch.stack(_x_true_valid).squeeze().to(device)
                    _x_true_in = torch.stack(_x_true_in).squeeze().to(device)

                    _state_goal_valid = _state_goal[:-model.c]
                    _goals_valid = [sg[0] for sg in _state_goal_valid]

                    _goals_valid = torch.stack(_goals_valid).squeeze().to(device)

                    _x_pred = mlp_deconv(_goals_valid.detach())

                    # save the figure
                    for i in range(_x_true_valid.shape[0]):
                        print('i:',i,'/',_x_true_valid.shape[0])
                        x_true = _x_true_valid[i].cpu().detach().numpy()
                        x_pred = _x_pred[i].cpu().detach().numpy()

                        x_true = np.transpose(x_true, (1, 2, 0))
                        x_pred = np.transpose(x_pred, (1, 2, 0))

                        plt.figure()
                        plt.imshow(x_true)
                        plt.savefig(os.path.join(os.path.join(dirname, f'epoch_{iii}'), f'x_true_{i}.png'))
                        # close
                        plt.close()

                        plt.figure()
                        plt.imshow(x_pred)
                        plt.savefig(os.path.join(os.path.join(dirname, f'epoch_{iii}'), f'x_pred_{i}.png'))
                        # close
                        plt.close()
                break_flag = True
            if break_flag:
                break
        if break_flag:
            break

    envs.close()
    #torch.save({
    #    'model': model.state_dict(),
    #    'args': args,
    #    'processor_mean': model.preprocessor.rms.mean,
    #    'optim': optimizer.state_dict()},
    #    f'models/{args.env_name}_{args.run_name}_steps={step}.pt')


def main(args):
    all_envs = gym.envs.registry.all()
    noframeskip_v4_no_ram_envs = [env.id for env in all_envs if
                                  ((env.id.endswith('NoFrameskip-v4')) and ('-ram' not in env.id) and ('Defender' not in env.id))]
    
    # make noframeskip_v4_no_ram_envs in reverse order
    noframeskip_v4_no_ram_envs = noframeskip_v4_no_ram_envs[::-1]


    run_name = args.run_name

    seeds_ = np.random.randint(-1000, 1000, 100)

    # runs = wandb.Api().runs("MDM_100k_collect_test")
    # existing_names = [run.name for run in runs]

    #for seed in range(len(noframeskip_v4_no_ram_envs)):
    for i in ['FuN']:
        for seed in range(len(noframeskip_v4_no_ram_envs)):
            # check if dirname = 'gen_goal/' + args.model_name + "_" + args.env_name exists
            dirname = 'gen_goal/' + i + "_" + noframeskip_v4_no_ram_envs[seed]
            if os.path.exists(dirname):
                continue

            env_name_ = noframeskip_v4_no_ram_envs[seed]

            args.model_name = i
            args.seed = seeds_[42]# random seed but for reproducibility
            args.env_name = env_name_

            experiment(args)
                


if __name__ == '__main__':
    main(args)
