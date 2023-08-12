import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize

from utils import init_hidden, weight_init
from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM
import numpy as np
import random
import gc
class HONET(nn.Module):
    def __init__(self,
                 num_workers,
                 input_dim,
                 hidden_dim_Hierarchies = [256, 256, 256, 256, 256],     # set of hidden_dims <- list form
                 time_horizon_Hierarchies = [1, 5, 15, 20, 25, 30],   #time_horizon & dilation -> time_horizon  # set of time_horizon <- list form
                 n_actions=17,
                 device='cuda',
                 dynamic=0,
                 args=None):

        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon_Hierarchies
        self.hidden_dim = hidden_dim_Hierarchies
        self.n_actions = n_actions
        self.dynamic = dynamic
        self.device = device
        self.eps = args.eps

        template_0 = torch.zeros(self.num_workers, self.hidden_dim[4])
        self.goal_0 = [torch.zeros_like(template_0).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        self.hierarchies_selected = torch.ones_like(torch.empty(self.num_workers, 3))

        self.preprocessor = Preprocessor(input_dim, device, mlp=False)
        self.percept = Perception(self.hidden_dim[-1],  self.time_horizon[0])
        self.policy_network = Policy_Network(self.hidden_dim[-1],  1000, num_workers)
        self.Hierarchy5_forth = Hierarchy5_forth(self.time_horizon[4], self.hidden_dim[4], args, device)
        self.Hierarchy4_forth = Hierarchy4_forth(self.time_horizon[3], self.hidden_dim[3], args, device)
        self.Hierarchy3_forth = Hierarchy3_forth(self.time_horizon[2], self.hidden_dim[2], args, device)
        self.Hierarchy2_forth = Hierarchy2_forth(self.time_horizon[1], self.hidden_dim[1], args, device)

        self.Hierarchy5_back = Hierarchy5_back(self.time_horizon[4], self.hidden_dim[4], args, device, num_workers)
        self.Hierarchy4_back = Hierarchy4_back(self.time_horizon[3], self.hidden_dim[3], args, device, num_workers)
        self.Hierarchy3_back = Hierarchy3_back(self.time_horizon[2], self.hidden_dim[2], args, device, num_workers)
        self.Hierarchy2_back = Hierarchy2_back(self.time_horizon[1], self.hidden_dim[1], args, device)

        self.goal_normalizer = Goal_Normalizer(self.hidden_dim[1])


        self.Hierarchy1 = Hierarchy1(self.num_workers, self.time_horizon[0], self.hidden_dim[4],  self.hidden_dim[1],  self.hidden_dim[0], self.n_actions, device)

        self.hidden_5 = init_hidden(args.num_workers, self.time_horizon[4] * self.hidden_dim[4],
                                    device=device, grad=True)
        self.hidden_4 = init_hidden(args.num_workers, self.time_horizon[3] * self.hidden_dim[3],
                                    device=device, grad=True)
        self.hidden_3 = init_hidden(args.num_workers, self.time_horizon[2] * self.hidden_dim[2],
                                    device=device, grad=True)
        self.hidden_2 = init_hidden(args.num_workers, self.time_horizon[1] * self.hidden_dim[1],
                                    device=device, grad=True)
        self.hidden_1 = init_hidden(args.num_workers, self.n_actions * self.hidden_dim[0],
                                    device=device, grad=True)
        self.hidden_percept = init_hidden(args.num_workers, self.time_horizon[0] * self.hidden_dim[-1],
                                    device=device, grad=True)
        self.hidden_policy_network = init_hidden(args.num_workers, 1000 * 4 * self.hidden_dim[1],
                                    device=device, grad=True)

        self.args = args
        self.to(device)
        self.apply(weight_init)

    def forward(self, x, goals_5, states_total, goals_4, goals_3, goals_2,  mask, step, train_eps, save=True):
        """A forward pass through the whole feudal network.

        Order of operations:
        1. input goes through a preprocessor to normalize and put on device
        2. normalized input goes to the perception module resulting in a state
        3. state is input for manager which produces a goal
        4. state and goal is both input for worker which produces an action
           distribution.

        Args:
            x (np.ndarray): observation from the environment
            goals (list):  list of goal tensors, length = 2 * r + 1
            states (list): list of state tensors, length = 2 * r + 1
            mask (tensor): mask describing for each worker if episode is done.
            save (bool, optional): If we are calculating next_v, we do not
                                   store rnn states. Defaults to True.
        """
        x = self.preprocessor(x)

        z = self.percept(x, self.hidden_percept, mask)

        goal_5_vanilla, hidden_5, value_5 = self.Hierarchy5_forth(z, self.hidden_5, mask)
        goal_4_vanilla, hidden_4, value_4 = self.Hierarchy4_forth(z, self.hidden_4, mask)
        goal_3_vanilla, hidden_3, value_3 = self.Hierarchy3_forth(z, self.hidden_3, mask)
        goal_2_vanilla, hidden_2, value_2 = self.Hierarchy2_forth(z, self.hidden_2, mask)


        goal_5_norm, goal_4_norm, goal_3_norm, goal_2_norm = self.goal_normalizer(goal_5_vanilla, goal_4_vanilla, goal_3_vanilla, goal_2_vanilla)

        if ((step % 1000) == 0):
            self.hierarchies_selected, hidden_policy_network = self.policy_network(z, goal_5_vanilla, goal_4_vanilla, goal_3_vanilla, self.hierarchies_selected, self.time_horizon, self.hidden_policy_network, mask, step)
            if (train_eps > torch.rand(1)[0]):
                self.hierarchies_selected[:, 0] = random.randrange(0,2)
                self.hierarchies_selected[:, 1] = random.randrange(0,2)
                self.hierarchies_selected[:, 2] = random.randrange(0,2)
            train_eps = train_eps * 0.99

        goal_5 = self.Hierarchy5_back(goal_5_norm, self.goal_0, self.hierarchies_selected[:, 0])
        goal_4 = self.Hierarchy4_back(goal_4_norm, goal_5, self.hierarchies_selected[:, 1])
        goal_3 = self.Hierarchy3_back(goal_3_norm, goal_4, self.hierarchies_selected[:, 2])
        goal_2 = self.Hierarchy2_back(goal_2_norm, goal_3)

        # Ensure that we only have a list of size 2*c + 1, and we use FiLo
        if len(goals_5) > (2 * self.time_horizon[4] + 1):
            goals_5.pop(0)
            states_total.pop(0)

        if len(goals_4) > (2 * self.time_horizon[3] + 1):
            goals_4.pop(0)

        if len(goals_3) > (2 * self.time_horizon[2] + 1):
            goals_3.pop(0)

        if len(goals_2) > (2 * self.time_horizon[1] + 1):
            goals_2.pop(0)

        goals_5.append(goal_5)
        goals_4.append(goal_4)
        goals_3.append(goal_3)
        goals_2.append(goal_2)
        states_total.append(z.detach())

        action_dist, hidden_1, value_1 = self.Hierarchy1(z, goals_2[:self.time_horizon[1] + 1], self.hidden_1, mask)

        if save:
            # Optional, don't do this for the next_v
            #self.hidden_percept = hidden_percept
            self.hidden_5 = hidden_5
            self.hidden_4 = hidden_4
            self.hidden_3 = hidden_3
            self.hidden_2 = hidden_2
            self.hidden_1 = hidden_1
            if ((step % 1000) == 0):
                self.hidden_policy_network = hidden_policy_network

        return action_dist, goals_5, states_total, value_5, goals_4, value_4, goals_3, value_3, goals_2, value_2, value_1, self.hierarchies_selected, train_eps
        #return action_dist, goals_5, states_total, goals_4, goals_3, goals_2, value_2, value_1, self.hierarchies_selected, train_eps

    def intrinsic_reward(self, states_2, goals_2, masks):
        return self.Hierarchy1.intrinsic_reward(states_2, goals_2, masks)

    def state_goal_cosine(self, states_n, goals_n, masks, hierarchy_num):
        if hierarchy_num == 5:
            return self.Hierarchy5_back.state_goal_cosine(states_n, goals_n, masks)
        if hierarchy_num == 4:
            return self.Hierarchy4_back.state_goal_cosine(states_n, goals_n, masks)
        if hierarchy_num == 3:
            return self.Hierarchy3_back.state_goal_cosine(states_n, goals_n, masks)
        if hierarchy_num == 2:
            return self.Hierarchy2_back.state_goal_cosine(states_n, goals_n, masks)

    def hierarchy_drop_reward(self, reward, hierarchy_selected):
        return self.policy_network.hierarchy_drop_reward(reward, hierarchy_selected)

    def repackage_hidden(self):
        def repackage_rnn(x):
            return [item.detach() for item in x]

        self.hidden_percept = repackage_rnn(self.hidden_percept)
        self.hidden_5 = repackage_rnn(self.hidden_5)
        self.hidden_4 = repackage_rnn(self.hidden_4)
        self.hidden_3 = repackage_rnn(self.hidden_3)
        self.hidden_2 = repackage_rnn(self.hidden_2)
        self.hidden_1 = repackage_rnn(self.hidden_1)

    def init_obj(self):
        template_5 = torch.zeros(self.num_workers, self.hidden_dim[4])
        template_4 = torch.zeros(self.num_workers, self.hidden_dim[3])
        template_3 = torch.zeros(self.num_workers, self.hidden_dim[2])
        template_2 = torch.zeros(self.num_workers, self.hidden_dim[1])
        goals_5 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        states_total = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        goals_4 = [torch.zeros_like(template_4).to(self.device) for _ in range(2 * self.time_horizon[3] + 1)]
        goals_3 = [torch.zeros_like(template_3).to(self.device) for _ in range(2 * self.time_horizon[2] + 1)]
        goals_2 = [torch.zeros_like(template_2).to(self.device) for _ in range(2 * self.time_horizon[1] + 1)]
        masks = [torch.ones(self.num_workers, 1).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        return goals_5, states_total, goals_4, goals_3, goals_2, masks

class Perception(nn.Module):
    def __init__(self, d, time_horizon):
        super().__init__()
        self.percept = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(32 * 9 * 9, d),
            nn.ReLU())

    def forward(self, x, hidden, mask):
        x1 = self.percept(x)
        return x1

class Policy_Network(nn.Module):
    def __init__(self, d, time_horizon, num_workers):
        super().__init__()
        self.Mrnn = DilatedLSTM(d*4, 3, time_horizon)
        self.num_workers = num_workers
    def forward(self, z, goal_5_norm, goal_4_norm, goal_3_norm, hierarchies_selected, time_horizon, hidden, mask, step):
        goal_x_info = torch.cat(([goal_5_norm.detach(), goal_4_norm.detach(), goal_3_norm.detach(), z]), dim=1)
        hidden = (mask * hidden[0], mask * hidden[1])
        policy_network_result, hidden = self.Mrnn(goal_x_info, hidden)
        policy_network_result = (policy_network_result - policy_network_result.detach().min(1, keepdim=True)[0]) / \
                                (policy_network_result.detach().max(1, keepdim=True)[0] - policy_network_result.detach().min(1, keepdim=True)[0])
        policy_network_result = policy_network_result.round()
        return policy_network_result.type(torch.int), hidden

    def hierarchy_drop_reward(self, reward, hierarchy_selected):
        #drop_reward = (reward - (hierarchy_selected.sum(dim=1).reshape(self.num_workers, 1))) / (reward+1)
        drop_reward = reward
        return drop_reward

class Goal_Normalizer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    def forward(self, goal_5, goal_4, goal_3, goal_2):
        minimum = min(goal_5.detach().min(), goal_4.detach().min(), goal_3.detach().min(), goal_2.detach().min())
        maximum = max(goal_5.detach().max(), goal_4.detach().max(), goal_3.detach().max(), goal_2.detach().max())
        goal_5_norm = (goal_5 - minimum) / (maximum - minimum)
        goal_4_norm = (goal_4 - minimum) / (maximum - minimum)
        goal_3_norm = (goal_3 - minimum) / (maximum - minimum)
        goal_2_norm = (goal_2 - minimum) / (maximum - minimum)
        return goal_5_norm, goal_4_norm, goal_3_norm, goal_2_norm


def Normalizer(goal):
    minimum = goal.detach().min()
    maximum = goal.detach().max()
    goal_normalized = (goal - minimum) / (maximum - minimum + 1e-9)
    return goal_normalized

class Hierarchy5_forth(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, hidden,  mask):
        hidden = (mask * hidden[0], mask * hidden[1])
        goal, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal)
        return goal, hidden, value_est

class Hierarchy5_back(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device, num_workers):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.num_workers = num_workers

    def forward(self, goal_norm, goal_up, hierarchies_selected):
        #goal_up = self.linear(goal_up)
        hierarchies_selected = hierarchies_selected.detach().reshape(self.num_workers, 1)
        goal_norm = hierarchies_selected.expand(self.num_workers, self.hidden_dim) * goal_norm
        goal = goal_norm #+ goal_up

        goal = Normalizer(goal)

        return goal

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy4_forth(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, hidden, mask):
        hidden = (mask * hidden[0], mask * hidden[1])
        goal, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal)

        return goal, hidden, value_est

class Hierarchy4_back(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device, num_workers):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.num_workers = num_workers

    def forward(self, goal_norm, goal_up, hierarchies_selected):
        goal_up = self.linear(goal_up.detach())
        hierarchies_selected = hierarchies_selected.detach().reshape(self.num_workers, 1)
        goal_norm = hierarchies_selected.expand(self.num_workers, self.hidden_dim) * goal_norm
        goal = goal_up + goal_norm

        goal = Normalizer(goal)

        return goal

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy3_forth(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, hidden, mask):
        hidden = (mask * hidden[0], mask * hidden[1])
        goal, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal)
        return goal, hidden, value_est

class Hierarchy3_back(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device, num_workers):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.num_workers = num_workers

    def forward(self, goal_norm, goal_up, hierarchies_selected):
        goal_up = self.linear(goal_up.detach())
        hierarchies_selected = hierarchies_selected.detach().reshape(self.num_workers, 1)
        goal_norm = hierarchies_selected.expand(self.num_workers, self.hidden_dim) * goal_norm
        goal = goal_up + goal_norm

        goal = Normalizer(goal)

        return goal

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy2_forth(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.space = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, hidden, mask):
        hidden = (mask * hidden[0], mask * hidden[1])
        goal, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal)
        return goal, hidden, value_est

class Hierarchy2_back(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.device = device
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, goal_norm, goal_up):
        goal_up = self.linear(goal_up.detach())
        goal = goal_up + goal_norm

        goal = Normalizer(goal)

        return goal
        #return goal_norm

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class critic_1(nn.Module):
    def __init__(self, hidden_dim_1, num_actions):
        self.hidden_dim_1 = hidden_dim_1
        self.num_actions = num_actions
        super().__init__()
        self.fc1 = nn.Linear(self.hidden_dim_1 * self.num_actions, 50)
        self.relu = nn.ReLU()
        self.out = nn.Linear(50, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.out(x).to('cpu')


class Hierarchy1(nn.Module):
    def __init__(self, num_workers, time_horizon, hiddne_dim_5, hidden_dim_2, hidden_dim_1, num_actions, device):
        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_actions = num_actions
        self.device = device

        self.Wrnn = nn.LSTMCell(self.hidden_dim_2, self.hidden_dim_1 * self.num_actions)
        self.phi = nn.Linear(self.hidden_dim_2, hidden_dim_1, bias=False)

        # self.critic = nn.Sequential(
        #     nn.Linear(self.hidden_dim_1 * self.num_actions, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 1)
        # )
        self.critic = critic_1(self.hidden_dim_1, self.num_actions)

    def forward(self, z, goals, hidden, mask):

        hidden = (mask * hidden[0].to(self.device), mask * hidden[1].to(self.device))
        u, cx = self.Wrnn(z, hidden)
        del hidden
        # Detaching is vital, no end to end training
        return torch.einsum("bk, bka -> ba", self.phi(torch.stack(goals).detach().sum(dim=0)), u.reshape(u.shape[0], self.hidden_dim_1, self.num_actions)).softmax(dim=-1).to('cpu'), (u.to('cpu'), cx.to('cpu')),  self.critic(u)

    def intrinsic_reward(self, states_s, goals_s, masks):

        t = self.time_horizon
        r_i = torch.zeros(self.num_workers, 1).to(self.device)
        mask = torch.ones(self.num_workers, 1).to(self.device)

        for i in range(1, self.time_horizon + 1):
            r_i_t = d_cos(states_s[t] - states_s[t - i], goals_s[t - i]).unsqueeze(-1)
            r_i = r_i + (mask * r_i_t)

            mask = mask * masks[t - i]

        r_i = r_i.detach()
        return r_i / self.time_horizon

def mp_loss(storage, next_v_5, next_v_4, next_v_3, next_v_2, next_v_1, args):

    # Discount rewards, both of size B x T
    ret_5 = next_v_5
    ret_4 = next_v_4
    ret_3 = next_v_3
    ret_2 = next_v_2
    ret_1 = next_v_1

    storage.placeholder()  # Fill ret_m, ret_w with empty vals
    for i in reversed(range(args.num_steps)):
        ret_5 = storage.r[i].to(args.device) + args.gamma_5 * ret_5 * storage.m[i].to(args.device)
        ret_4 = storage.r[i].to(args.device) + args.gamma_4 * ret_4 * storage.m[i].to(args.device)
        ret_3 = storage.r[i].to(args.device) + args.gamma_3 * ret_3 * storage.m[i].to(args.device)
        ret_2 = storage.r[i].to(args.device) + args.gamma_2 * ret_2 * storage.m[i].to(args.device)
        ret_1 = storage.r[i].to(args.device) + args.gamma_1 * ret_1.to(args.device) * storage.m[i].to(args.device)
        storage.ret_5[i] = ret_5
        storage.ret_4[i] = ret_4
        storage.ret_3[i] = ret_3
        storage.ret_2[i] = ret_2
        storage.ret_1[i] = ret_1

    # Optionally, normalize the returns
    storage.normalize(['ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1'])

    rewards_intrinsic, value_5, value_4, value_3, value_2, value_1, ret_5, ret_4, ret_3, ret_2, ret_1, logps, entropy, \
        state_goal_5_cosines, state_goal_4_cosines, state_goal_3_cosines, state_goal_2_cosines, hierarchy_selected, hierarchy_drop_reward = storage.stack(
        ['r_i', 'v_5', 'v_4', 'v_3', 'v_2', 'v_1', 'ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1',
         'logp', 'entropy', 'state_goal_5_cos', 'state_goal_4_cos', 'state_goal_3_cos', 'state_goal_2_cos', 'hierarchy_selected', 'hierarchy_drop_reward'])


    rewards_intrinsic, value_5, value_4, value_3, value_2, value_1, ret_5, ret_4, ret_3, ret_2, ret_1, logps, entropy, \
        state_goal_5_cosines, state_goal_4_cosines, state_goal_3_cosines, state_goal_2_cosines, hierarchy_selected, hierarchy_drop_reward = \
    rewards_intrinsic.to(args.device), value_5.to(args.device), value_4.to(args.device), value_3.to(args.device), value_2.to(args.device), value_1.to(args.device), ret_5.to(args.device), \
        ret_4.to(args.device), ret_3.to(args.device), ret_2.to(args.device), ret_1.to(args.device), logps.to(args.device), entropy.to(args.device), \
        state_goal_5_cosines.to(args.device), state_goal_4_cosines.to(args.device), state_goal_3_cosines.to(args.device), state_goal_2_cosines.to(args.device), hierarchy_selected.to(args.device), hierarchy_drop_reward.to(args.device)


    advantage_5 = ret_5 - value_5
    loss_5 = (state_goal_5_cosines * advantage_5.detach()).mean()
    hierarchy_selected_5 = hierarchy_selected[:, :, 0].float().mean()
    value_5_loss = 0.5 * advantage_5.pow(2).mean()

    advantage_4 = ret_4 - value_4
    loss_4 = (state_goal_4_cosines * advantage_4.detach()).mean()
    hierarchy_selected_4 = hierarchy_selected[:, :, 1].float().mean()
    value_4_loss = 0.5 * advantage_4.pow(2).mean()

    advantage_3 = ret_3 - value_3
    loss_3 = (state_goal_3_cosines * advantage_3.detach()).mean()
    hierarchy_selected_3 = hierarchy_selected[:, :, 2].float().mean()
    value_3_loss = 0.5 * advantage_3.pow(2).mean()

    advantage_2 = ret_2 - value_2
    loss_2 = (state_goal_2_cosines * advantage_2.detach()).mean()
    value_2_loss = 0.5 * advantage_2.pow(2).mean()

    # Calculate advantages, size B x T
    advantage_1 = ret_1 + args.alpha * rewards_intrinsic - value_1
    loss_1 = (logps * advantage_1.detach()).mean()
    value_1_loss = 0.5 * advantage_1.pow(2).mean()

    entropy = entropy.mean()

    hierarchy_drop_reward = hierarchy_drop_reward.mean()

    loss = (- loss_5 - loss_4 - loss_3 - loss_2 - loss_1 + value_5_loss + value_4_loss + value_3_loss + value_2_loss + value_1_loss - hierarchy_drop_reward) - args.entropy_coef * entropy
    #loss = (- loss_2 - loss_1 +  value_2_loss + value_1_loss) - args.entropy_coef * entropy

    return loss, {'loss/total_mp_loss': loss.item(),
                  'loss/Hierarchy_5': loss_5.item(),
                  'loss/Hierarchy_4': loss_4.item(),
                  'loss/Hierarchy_3': loss_3.item(),
                  'loss/Hierarchy_2': loss_2.item(),
                  'loss/Hierarchy_1': loss_1.item(),

                  'loss/value_Hierarchy_5': value_5_loss.item(),
                  'loss/value_Hierarchy_4': value_4_loss.item(),
                  'loss/value_Hierarchy_3': value_3_loss.item(),
                  'loss/value_Hierarchy_2': value_2_loss.item(),
                  'loss/value_Hierarchy_1': value_1_loss.item(),

                  'hierarchy_use/hierarchy_selected_5': hierarchy_selected_5.item(),
                  'hierarchy_use/hierarchy_selected_4': hierarchy_selected_4.item(),
                  'hierarchy_use/hierarchy_selected_3': hierarchy_selected_3.item(),

                  'policy_network' : hierarchy_drop_reward,

                  'value_Hierarchy_1/entropy': entropy.item(),
                  'value_Hierarchy_1/advantage': advantage_1.mean().item(),
                  'value_Hierarchy_1/intrinsic_reward': rewards_intrinsic.mean().item(),

                  'value_Hierarchy_2/cosines': state_goal_2_cosines.mean().item(),
                  'value_Hierarchy_2/advantage': advantage_2.mean().item(),

                  'value_Hierarchy_3/cosines': state_goal_3_cosines.mean().item(),
                  'value_Hierarchy_3/advantage': advantage_3.mean().item(),

                  'Hierarchy_4/cosines': state_goal_4_cosines.mean().item(),
                  'Hierarchy_4/advantage': advantage_4.mean().item(),

                  'Hierarchy_5/cosines': state_goal_5_cosines.mean().item(),
                  'Hierarchy_5/advantage': advantage_5.mean().item()
                  }