#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : vissim_v0.py
# @Note    : vissim simulation environment of version 0
"""
Note: 通过 pip install gym 安装gym工具包
      复制整个vistraci文件夹到 xx/Lib/site-packages/gym/envs下
      在 `xx/Lib/site-packages/gym/envs/__init__.py` 中注册vissim环境,在该文件最后加入以下代码
        register(
            id='Vissim-v0',
            entry_point='gym.envs.vistraci:VissimEnv0',
            max_episode_steps=10**4,
            reward_threshold=100.0,
        )
"""
import logging
import time

import gym
import numpy as np

from .vistraci import VissimComServer

logger = logging.getLogger(__name__)


# Gym style call environment
class VissimEnv0(gym.Env):
    """
    Advanced API package for Gym platform:
        1、reset(self)：重置环境的状态,返回observation
        2、step(self, action)：选择动作编号对应的配时方案运行一个周期,返回observation, reward, done, info
        3、render(self)：显示实时仿真画面,无返回值
        4、close(self)：关闭环境,并清除内存,无返回值
        5、seed(self)：设置环境运行随机数,无返回值
        note：需要注意的是,render需要在reset和step之前调用才能生效,这一点与gym不同
    """
    metadata = {
        'render.modes': ['3D', '2D', 'accelerate'],
    }

    def __init__(self):
        self.com = None
        self.action_space = None
        self.observation_space = None
        self.state_dim = None
        self.plans = None
        self.states = None
        self.flow_flag = True  # vehicle flow detection flag
        self.speed_flag = True  # vehicle speed detection flag
        self.queue_flag = True  # vehicle queue detection flag
        self.net_file = ""

    # Define action space
    def define_action_space(self, plans):
        """
        Args:
            plans: An array of timing plan, detailed parameters are as follows:
                   [cycle_time, amber_time, clear_time, green_time]
                   cycle_time: The signal cycle time of intersection
                   amber_time: The yellow light time of each phase
                   clear_time: Red light clearance time of each phase
                   green_time: Green time of each phase
                   An example is as follows:
                   [
                        [170, [3, 3, 3], [2, 2, 2], [43, 55, 57]],
                        [170, [3, 3, 3], [2, 2, 2], [49, 51, 55]],
                        ......
                   ]
                   instruction: (3+2+43) + (3+2+55) + (3+2+57) = 170
        Returns: No return value
        """
        self.plans = plans
        self.action_space = gym.spaces.Discrete(len(self.plans))  # 定义离散动作空间

    # Defining state space
    def define_observation_space(self, states, min_value=0, max_value=1000):
        self.states = states
        self.state_dim = len(states)
        low = np.array([min_value for _ in range(self.state_dim)], dtype=np.float32)
        high = np.array([max_value for _ in range(self.state_dim)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)  # 定义连续状态空间

    # Loading network Files
    def load_net(self, file):
        self.net_file = file

    # Activated detector
    def activate_detector(self, flow_flag=True, speed_flag=True, queue_flag=True):
        self.flow_flag = flow_flag
        self.speed_flag = speed_flag
        self.queue_flag = queue_flag

    # Reset the environment and start the simulation
    def reset(self):
        """
        Args: None input parameter
        Returns:
            observation: list, traffic state list, flow + speed + queue
                flow: sum vehicle flow of signal cycle time
                speed: average speed of signal cycle time
                queue: average queue of signal cycle time
        """
        self.com = VissimComServer()
        if self.net_file == "":
            raise ValueError("No path is specified for the road network file in *.inp")
        else:
            self.com.load_net(self.net_file)
            self.com.start()
            flow = [0 for _ in range(self.com.data_collections.Count)] if self.flow_flag else []
            speed = [0 for _ in range(self.com.data_collections.Count)] if self.speed_flag else []
            queue = [0 for _ in range(self.com.queue_counters.Count)] if self.queue_flag else []
            observation = flow + speed + queue
            return observation

    # The simulation runs for one cycle
    def step(self, action):
        """
        Args:
            action: number of actions performed by an agent
        Returns:
            observation: list, traffic state list, [vehicle flow, average speed, average queue]
            reward: float, delay time of intersection, time
            done: bool, end of simulation flag, True or False
            info: str, some simulation information, such as:
                  system time、simulation time、flow、speed、queue、delay time、action
        """
        plan = self.plans[action]
        cycle_time = plan[0]
        if cycle_time <= 0:
            raise ValueError("Error, the cycle time cannot be less than or equal to 0\n")
        else:
            self.com.sc[0].SetAttValue("CYCLETIME", cycle_time)
            self.com.control_signal_group(plan[1], plan[2], plan[3])
            if cycle_time is None:
                cycle_time = self.com.sc[0].AttValue("CYCLETIME")
            last_elapsed_time = self.com.simulation.AttValue("ELAPSEDTIME")
            stop_time = cycle_time + last_elapsed_time
            while True:
                elapsed_time = self.com.simulation.AttValue("ELAPSEDTIME")
                if stop_time > elapsed_time >= 0:
                    self.com.simulation.RunSingleStep()
                else:
                    self.com.ct = self.com.sc[0].AttValue("CYCLETIME")
                    self.com.offset = self.com.sc[0].AttValue("OFFSET")
                    self.com.elapsed_time = self.com.simulation.AttValue("ELAPSEDTIME")
                    break
            # At the end of the current signal cycle, the traffic detector data in the past signal cycle is read
            flow = list(map(lambda x: int(x), self.com.get_flow_collections_detector())) if self.flow_flag else []
            speed = list(
                map(lambda x: round(x, 2), self.com.get_speed_collections_detector())) if self.speed_flag else []
            queue = list(
                map(lambda x: round(x, 2), self.com.get_queue_counters_detector())) if self.queue_flag else []
            delay = list(map(lambda x: round(x, 3), self.com.get_delay_times_detector()))
            observation = flow + speed + queue
            reward = delay[0]
            if self.com.elapsed_time >= self.com.simulation_stop_time - cycle_time:
                done = True
            else:
                done = False
            sys_time = time.strftime("%H:%M:%S")
            sim_time = int(self.com.elapsed_time)
            info = "%-8s  %-6s  %-32s %-56s %-52s %-7s %-4s" % \
                   (str(sys_time), str(sim_time), str(flow), str(speed), str(queue), str(delay), str(action))
            return observation, reward, done, info

    # Visual visSIM real-time simulation screen
    def render(self, mode="2D"):
        if mode == '3D':
            self.com.graphics.SetAttValue("3D", True)
            self.com.graphics.SetAttValue("VISUALIZATION", True)
        elif mode == '2D':
            self.com.graphics.SetAttValue("3D", False)
            self.com.graphics.SetAttValue("VISUALIZATION", True)
        elif mode == 'accelerate':
            self.com.graphics.SetAttValue("VISUALIZATION", False)
        else:
            super(VissimEnv0, self).render(mode=mode)  # just raise an exception

    # Close the environment, exit the visSIM simulation, and clear the memory
    def close(self):
        self.com.end()

    # Set the random number seed
    def seed(self, seed=42):
        if seed < 0:
            raise ValueError("The seed of random number cannot be less than 0")
        else:
            self.com.simulation.RandomSeed = seed
