#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : .demo.main.py
# @Note    : This is a sample of python script whitch use vistraci module to control traffic signal.

import os
from itertools import permutations, product, combinations_with_replacement

import gym

net = ["net13", "net8"]
index = 1

net_path = os.getcwd().replace("\\", "/") + "/net/%s/net.inp" % net[index]
plan_path = os.getcwd().replace("\\", "/") + "/plan/%s_plan.txt" % net[index]


# 创建配时方案
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
    loss = sum(amber) + sum(clear)
    try:
        with open(file, "a+") as f:
            f.truncate(0)
    except (IndexError, Exception):
        pass
    plans = []
    unused_plan = 0
    for i in range(len(green)):
        if sum(green[i]) == cycle - loss:
            if num <= 0 or len(plans) < num:
                plan = [cycle, amber, clear, list(green[i])]
                plans.append(plan)
                with open(file, "a+") as f:
                    f.write("%s\n" % (str(plan)))
            else:
                unused_plan += 1
    print("save plan file to %s" % file)
    return plans, unused_plan


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


if __name__ == '__main__':
    # make dirs
    os.makedirs('./plan/', exist_ok=True)

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
    env = gym.make("Vissim-v0")  # create train environment
    env.load_net(net_path)  # load simulation road net
    env.define_action_space(signal_plans)  # define environment action space

    observation = env.reset()
    for epoch in range(200):
        # env.render("accelerate")  # 关闭仿真画面, 可加速仿真速度
        # env.render("2D")  # 每回合开始重新打开2D平面仿真画面
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("---------------------------------------------------------")
        print("epoch: %d, 观测状态：%s" % (epoch, observation))
        print("epoch: %d, 延误时间: %.3f" % (epoch, reward))
        print("---------------------------------------------------------")
        if done:
            break
    env.close()
