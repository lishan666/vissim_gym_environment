#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : environment.vissim.py
# @Note    : This is a sample of python script whitch use vistraci module to control traffic signal.

import os
from itertools import permutations, product, combinations_with_replacement

import gym
import numpy as np
from torch import cuda, load

from agent import dqn

net = ["net13", "net8"]
index = 1

net_path = os.getcwd().replace("\\", "/") + "/net/%s/net.inp" % net[index]
plan_path = os.getcwd().replace("\\", "/") + "/plan/%s_plan.txt" % net[index]

device = 'cuda' if cuda.is_available() else 'cpu'

EPISODE = 2000
WARM = 5
STEP = 42
ZEROREWARD = 38.509
CONVERGENCE_UP = 4  # 奖励收敛上限  (range:-10~10)
CONVERGENCE_LOW = -4  # 奖励收敛下限  (range:-10~10)
CONVERGENCE = int(STEP * 0.20) + 1  # 收敛计数器
MEMORY_CAPACITY = STEP * 10  # 存储池容量
ALGORITHM = "DQN"

EPSILON_MAX = 0.99  # greedy 最大贪婪策略
EPSILON_MIN = 0.05  # greedy 最小贪婪策略
LR_MAX = 0.01  # 最大学习率
LR_MIN = 1e-5  # 最小学习率
MAX_STEP = 42  # 最大回合步长
TEST_STEP = MAX_STEP  # 测试步长
TEST_FREQUENCY = 10 / EPISODE  # 测试频率

# 超参数
LR = LR_MAX  # 学习率
EPSILON = EPSILON_MAX  # greedy 贪婪策略
GAMMA = 0.95  # 奖赏折扣
BATCH_SIZE = 32  # 批处理大小
UPDATE_STEP = 50  # 目标网络更新步长
EPSILON_DAMPING = (EPSILON_MIN / EPSILON) ** (1 / EPISODE)  # 探索衰减因子
LR_DAMPING = (LR_MIN / LR) ** (1 / EPISODE)  # 学习衰减因子

N_ACTIONS = 20
N_STATES = 24
ENV_A_SHAPE = 0
NODE = 100

LOSS = "SmoothL1Loss"
OPTIM = "Adam"
ACTIVATE = "relu"

plan_file = "./fix/txt/test_fix_plans_define.txt"
train_file = './model/txt/train_record.txt'
train_test_file = './model/txt/test_record.txt'
drl_test_file = './model/txt/test_drl_record.txt'
status_file = './model/txt/test_status_record.txt'
online_network = './model/pkl/online_network_best.pkl'
target_network = './model/pkl/target_network_best.pkl'
test_network = './model/pkl/online_network_best.pkl'

hint = 1
model = 2
strategy_fun = 2
damping_fun = 3
reward_fun = 3


# create signal timing plans
def create_plans(cycle=None,
                 green=None,
                 amber=None,
                 clear=None,
                 num=0,
                 file="./plan.txt", ):
    """
    Args:
        cycle: signal cycle
        green: Green time of each phase
        amber: The yellow light time of each phase
        clear: Red light clearance time of each phase
        num: Max number of signal timing plans, 0 indicates that no number is specified
        file: Path to save the signal timing plan
    Example:
        create_plans(cycle=170,
                    green=[41,43......,57],
                    amber=[3,3,3],
                    clear=[2,2,2],
                    num=20,
                    file="plan.txt",)
    Returns:

    """
    if cycle is None:
        raise ValueError("Cycle signal time error")
    if green is None:
        raise ValueError("Green signal time error")
    if amber is None:
        raise ValueError("Amber signal time error")
    if clear is None:
        raise ValueError("Clean red signal time error")
    if not (len(amber) == len(clear) == len(green[0])):
        raise ValueError("The phase number of each color signal lamp is contradictory")
    loss_time = sum(amber) + sum(clear)
    try:
        with open(file, "a+") as f:
            f.truncate(0)
    except (IndexError, Exception):
        pass
    plans = []
    unused_plan = 0
    for i in range(len(green)):
        if sum(green[i]) == cycle - loss_time:
            if num <= 0 or len(plans) < num:
                plan = [cycle, amber, clear, list(green[i])]
                plans.append(plan)
                with open(file, "a+") as f:
                    f.write("%s\n" % (str(plan)))
            else:
                unused_plan += 1
    print("save plan file to %s" % file)
    return plans, unused_plan


# 奖赏函数2
def get_reward_2(delay, zeroreward):
    return round(zeroreward - delay, 3)


# CPU电源模式配置(默认开启高性能模式,可加速)
def power_config(idx=2):
    import subprocess
    mode = [
        "a1841308-3541-4fab-bc81-f71556f20b4a",  # 节能
        "381b4222-f694-41f0-9685-ff5bb260df2e",  # 平衡
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",  # 高性能
        "e9a42b02-d5df-448d-aa00-03f14749eb61",  # 卓越
    ]
    subprocess.call("Powercfg -s %s" % mode[idx])


# train agent
def train(env, agent):
    # train stage
    best_reward = -10
    for episode in range(EPISODE):
        success = 0
        max_sucess = 0
        fail = 0
        loss = 0
        step_count = 0
        convergence_test = False
        delay_record = []
        rewrad_record = []
        loss_record = []
        # warm-up
        # env.render("accelerate")  # 关闭仿真画面, 可加速仿真速度
        for i in range(WARM):
            state, _, _, _ = env.step(0)
        # formal train
        state, _, _, _ = env.step(0)
        for step in range(STEP):
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            redefine_reward = get_reward_2(reward, ZEROREWARD)
            # judgment convergence condition
            if redefine_reward >= CONVERGENCE_UP:
                success += 1
                if success > max_sucess:
                    max_sucess = success
            else:
                success = 0
            if success >= CONVERGENCE:
                convergence_test = True
                done = True
            if redefine_reward <= CONVERGENCE_LOW:
                fail += 1
            else:
                fail = 0
            if fail >= CONVERGENCE:
                done = True
            # store samples to the experience pool
            agent.store(state, action, redefine_reward, next_state, done)
            # 智能体进行学习
            if agent.memory_counter > MEMORY_CAPACITY:
                loss = agent.learn(ALGORITHM)
                # 保存误差信息
                with open('./model/txt/loss.txt', 'a+') as f:
                    f.write(str(float(loss)) + ',')
            # 更新状态、奖励、平均延误、当前回合训练步数
            state = next_state
            loss_record.append(loss)
            delay_record.append(reward)
            rewrad_record.append(redefine_reward)
            step_count += 1
            # 判断当前回合仿真结束标志
            if done:
                break
        # 输出并保存当前回合数、回合总步数、探索概率、学习率、平均奖励、平均延误、平均损失、最大收敛次数
        mean_delay = sum(delay_record) / len(delay_record)
        mean_reward = sum(rewrad_record) / len(rewrad_record)
        mean_loss = sum(loss_record) / len(loss_record)
        # 保存训练回合记录文件
        with open(train_file, 'a+') as f:
            record = "%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t\n" % \
                     (str(episode), str(step_count), str(round(dqn.EPSILON, 3)), str(round(dqn.LR, 3)),
                      str(max_sucess),
                      str(round(mean_delay, 3)),
                      str(round(mean_reward, 3)),
                      str(round(mean_loss, 3)))
            f.write(record)
        info = 'episode: {}, train_mean_delay: {}, train_mean_reward: {}, train_mean_loss: {}'. \
            format(episode, round(mean_delay, 3), round(mean_reward, 3), round(mean_loss, 3))
        print(info)
        # 逐渐衰减探索率s
        if dqn.EPSILON > EPSILON_MIN:
            # 衰减方法一  指数衰减
            # ag.EPSILON *= EPSILON_DAMPING
            # 衰减方法二  余弦衰减
            x = episode / EPISODE * np.pi
            y = EPSILON_MIN + (np.cos(x) + 1) / 2 * (EPSILON_MAX - EPSILON_MIN)
            dqn.EPSILON = y

        # Validation network model Performance
        c1 = convergence_test
        c2 = (episode % max(1, int(EPISODE * TEST_FREQUENCY)) == 0)
        c3 = (episode == EPISODE)
        c4 = (episode == 1)
        if c1 or c2 or c3 or c4:
            # 初始化参数
            test_delay_record = []
            test_rewrad_record = []
            # 重启环境并获取初始交通流状态
            state = env.reset()
            # 热身时间
            for i in range(5):
                state, reward, done, info = env.step(0)
            # 运行指定个仿真周期
            for step in range(TEST_STEP):
                # 由交通流状态获取配时动作方案
                action = agent.action(state, random=False)
                # vissim环境采取动作运行一周期，得到下一周期的状态信息
                next_state, reward, done, info = env.step(action)
                # 重定义奖励
                delay = reward
                redefine_reward = get_reward_2(delay, ZEROREWARD)
                test_delay_record.append(delay)
                test_rewrad_record.append(redefine_reward)
                # 更新状态
                state = next_state
            # 输出当前回合、测试平均延误、测试平均奖励
            test_mean_delay = sum(test_delay_record) / len(test_delay_record)
            test_mean_reward = sum(test_rewrad_record) / len(test_rewrad_record)
            info = 'episode: {}, test_mean_delay: {}, test_mean_reward: {}'. \
                format(episode, round(test_mean_delay, 3), round(test_mean_reward, 3))
            print(info)
            # 保存测试回合记录文件
            with open(train_test_file, 'a+') as f:
                record = "%-5s\t%-5s\t%-5s\t%-5s\t\n" % \
                         (str(episode), str(TEST_STEP),
                          str(round(test_mean_delay, 3)),
                          str(round(test_mean_reward, 3)))
                f.write(record)
            # 保存历史训练最优网络模型
            if test_mean_reward > best_reward:
                best_reward = test_mean_reward
                agent.save(episode, './model/pkl/')
    env.close()


# test agent
def test(env, agent, online_net, target_net):
    # 查看网络大小
    net_data = load(online_net, map_location=device)
    for key, value in net_data.items():
        print(key, value.size())
    print(agent.online_net)

    # 加载目标网络
    agent.online_net.load_state_dict(load(online_net, map_location=device))
    agent.target_net.load_state_dict(load(target_net, map_location=device))
    # agent decision
    for episode in range(10):
        state = env.reset()
        for step in range(10):
            print(f"episode: {episode}, step: {step}")
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            redefine_reward = get_reward_2(reward, ZEROREWARD)
            print("---------------------------------------------------------")
            print("episode: %d, epoch: %d, 当前状态：%s" % (episode, step, state))
            print("episode: %d, epoch: %d, 执行动作：%s" % (episode, step, action))
            print("episode: %d, epoch: %d, 延误时间: %.3f" % (episode, step, reward))
            print("episode: %d, epoch: %d, 转移状态：%s" % (episode, step, next_state))
            print("episode: %d, epoch: %d, 奖赏值: %.3f" % (episode, step, redefine_reward))
            print("---------------------------------------------------------")
            # 更新状态、奖励、平均延误、当前回合训练步数
            state = next_state
            # 判断当前回合仿真结束标志
            if done:
                break
    env.close()


if __name__ == '__main__':
    # make dirs
    path_list = ['./model', './model/pkl', './model/txt', './model/png', './backup', 'best', './plan/']
    for path in path_list:
        os.makedirs(path, exist_ok=True)

    # power config
    power_config()

    # create timing plan
    if net[index] == "net13":
        green_low, green_high, green_interval, phase_num = 41, 57, 2, 3
    elif net[index] == "net8":
        green_low, green_high, green_interval, phase_num = 26, 32, 2, 4
    else:
        green_low, green_high, green_interval, phase_num = 41, 57, 2, 3
    choose_space = range(green_low, green_high + 1, green_interval)
    replacement = list(combinations_with_replacement(choose_space, phase_num))  # 可重复,增大
    product = list(product(choose_space, repeat=phase_num))  # 可重复,无序
    permutation = list(permutations(choose_space, phase_num))  # 不可重复
    if net[index] == "net13":
        signal_plans, _ = create_plans(cycle=170,
                                       green=permutation,
                                       amber=[3, 3, 3],
                                       clear=[2, 2, 2],
                                       num=20,
                                       file=plan_path, )
    elif net[index] == "net8":
        signal_plans, _ = create_plans(cycle=130,
                                       green=product,
                                       amber=[3, 3, 3, 3],
                                       clear=[2, 2, 2, 2],
                                       num=20,
                                       file=plan_path, )
    else:
        signal_plans, _ = create_plans(cycle=170,
                                       green=permutation,
                                       amber=[3, 3, 3],
                                       clear=[2, 2, 2],
                                       num=20,
                                       file=plan_path, )

    # define environment
    Env = gym.make("Vissim-v0")  # create train environment
    Env.load_net(net_path)  # load simulation road net
    Env.define_action_space(signal_plans)  # define environment action space

    # define agent
    dqn.N_ACTIONS = len(signal_plans)
    dqn.N_STATES = len(Env.reset())
    Agent = dqn.Agent()

    # train agent
    # train(Env, Agent)

    # test agent
    online_network = "./best/%s/online_network_best.pkl" % net[index]
    target_network = "./best/%s/target_network_best.pkl" % net[index]
    test(Env, Agent, online_network, target_network)
