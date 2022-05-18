#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 14:38
# @Author  : LiShan
# @Email   : lishan_1997@126.com
# @File    : dqn.py
# @Note    : this is dqn algorithm and network model
"""
refer：
[1]https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py
[2]https://github.com/GAOYANGAU/DRLPytorch/blob/master/nature-DQN.py
[3]https://github.com/catziyan/DRLPytorch-/tree/master/08
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun

device = "cpu"
# 超参数
BATCH_SIZE = 32  # 批处理大小
LR = 0.01  # 学习率
EPSILON = 0.95  # greedy 贪婪策略
GAMMA = 0.95  # 奖赏折扣
UPDATE_STEP = 50  # 目标网络更新步长
MEMORY_CAPACITY = 500  # 存储池容量
LR_MIN = 1e-5  # 最小学习率

# 状态动作、空间
N_STATES = 39  # 状态空间维度
N_ACTIONS = 20  # 动作空间大小
ENV_A_SHAPE = 0

# 隐藏层结点数
NODE = 100


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, NODE)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(NODE, NODE)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(NODE, N_ACTIONS)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # h1 = fun.leaky_relu(self.fc1(x))
        # h2 = fun.leaky_relu(self.fc2(h1))
        h1 = fun.relu(self.fc1(x))
        h2 = fun.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class Agent:
    def __init__(self):
        self.online_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2 + 1))
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()

    def action(self, state, random=True):
        # 行方向扩充维度
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)
        random_num = np.random.uniform()
        # epsilon greedy 贪婪策略 探索  随机选择动作
        if random and (random_num < EPSILON):
            action = np.random.randint(0, N_ACTIONS)
        # epsilon greedy 贪婪策略 利用  选择Q估计值最大的动作
        else:
            # 利用在线网络求当前状态的动作价值
            actions_value = self.online_net.forward(state.to(device))
            # 求动作价值中，每行最大值的索引
            action = torch.max(actions_value, 1)[1].data.to('cpu').numpy()[0]
        action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store(self, state, action, reward, next_state, done):
        if done:
            transition = np.hstack((state, action, reward, next_state, 0))
        else:
            transition = np.hstack((state, action, reward, next_state, 1))
        # 用新内存值替换旧内存值
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, algorithm="DQN"):
        # # 更新学习率
        # self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)
        # 目标网络参数更新
        if self.learn_step_counter % UPDATE_STEP == 0:
            self.target_net.load_state_dict(OrderedDict(self.online_net.state_dict()))
        self.learn_step_counter += 1

        # 小批量采样
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :N_STATES]).to(device)
        batch_action = torch.LongTensor(batch_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, N_STATES + 1:N_STATES + 2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -N_STATES - 1:-1]).to(device)
        batch_done = torch.FloatTensor(batch_memory[:, -1]).to(device)

        # Q 值更新算法
        if algorithm == "DQN":
            # 求当前状态Q估计值，并按batch_action的列索引得到对应动作的Q值
            q_eval = self.online_net.forward(batch_state).gather(1, batch_action)
            # 求下一状态Q目标值，反向传播时不计算导数，即不更新目标网络
            q_next = self.target_net.forward(batch_next_state).detach()
            # 求目标Q值
            q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * batch_done.view(BATCH_SIZE, 1)
            # 反向传播和更新网络参数
            loss = self.loss_func(q_eval, q_target)
            loss_value = float(loss)
            # 设置初始梯度为0
            self.optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 更新权重
            self.optimizer.step()
            return loss_value

        elif algorithm == "DDQN":
            actions_value = self.online_net.forward(batch_next_state)
            next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
            q_eval = self.online_net.forward(batch_state).gather(1, batch_action)
            q_next = self.target_net.forward(batch_next_state).gather(1, next_action)
            q_target = batch_reward + GAMMA * q_next * batch_done.view(BATCH_SIZE, 1)
            # 反向传播和更新网络参数
            loss = self.loss_func(q_eval, q_target)
            loss_value = float(loss)
            # 设置初始梯度为0
            self.optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 更新权重
            self.optimizer.step()
            return loss_value

    def save(self, episode, file="./", best=True):
        torch.save(self.online_net.state_dict(), file + 'online_network_%d.pkl' % episode)
        torch.save(self.target_net.state_dict(), file + 'target_network_%d.pkl' % episode)
        if best:
            torch.save(self.online_net.state_dict(), file + 'online_network_best.pkl')
            torch.save(self.target_net.state_dict(), file + 'target_network_best.pkl')
