#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : vistraci.py
# @Note    : vissim traffic control interface
from random import randint

import psutil
from win32com.client import Dispatch


# Vissim component object mode server
class VissimComServer:
    def __init__(self):
        """get vissim component object mode interface and initialize object"""
        if not self.detect_vissim():
            raise EnvironmentError("Vissim is not running, please check!\n")
        self.vissim = Dispatch("VISSIM.Vissim")
        self.net_file = ""
        self.simulation = None
        self.graphics = None
        self.net = None
        self.links = None
        self.inputs = None
        self.vehicles = None
        self.controllers = None
        self.groups = None
        self.data_collections = None
        self.travel_times = None
        self.delays = None
        self.queue_counters = None
        # 评价接口
        self.eval = None
        self.qceval = None
        self.dceval = None
        self.deval = None
        self.tteval = None
        self.linkeval = None
        # 信号控制机、信号灯组
        self.sc, self.sg = [], []
        # 检测器、检测器保存数据
        self.tt, self.travel_time = [], []
        self.dt, self.delay = [], []
        self.dc, self.vel, self.speed = [], [], []
        self.qc, self.queue_length = [], []
        # 车流输入
        self.ip, self.ip_flow = [], []
        # 信号周期、相位差（周期延迟时间）、运行时间
        self.ct = 0
        self.offset = 0
        self.elapsed_time = 0
        # 仿真停止时间
        self.simulation_stop_time = 999999

    def connect(self):
        """connect VISSIM_COMServer"""
        self.simulation = self.vissim.Simulation
        self.graphics = self.vissim.Graphics
        self.net = self.vissim.net
        self.links = self.net.Links
        self.inputs = self.net.VehicleInputs
        self.vehicles = self.net.Vehicles
        self.controllers = self.net.SignalControllers
        self.groups = self.controllers(1).SignalGroups
        self.data_collections = self.net.DataCollections
        self.travel_times = self.net.TravelTimes
        self.delays = self.net.Delays
        self.queue_counters = self.net.QueueCounters
        # 评价接口
        self.eval = self.vissim.Evaluation
        self.qceval = self.eval.QueueCounterEvaluation
        self.dceval = self.eval.DataCollectionEvaluation
        self.deval = self.eval.DelayEvaluation
        self.tteval = self.eval.TravelTimeEvaluation
        self.linkeval = self.eval.LinkEvaluation
        # 打开检测器评价接口
        self.eval.SetAttValue("DataCollection", True)
        self.eval.SetAttValue("TRAVELTIME", True)
        self.eval.SetAttValue("DELAY", True)
        self.eval.SetAttValue("QUEUECOUNTER", True)
        # self.eval.SetAttValue("LINK", True)
        # 打开检测器文件评价接口
        self.qceval.SetAttValue("FILE", True)
        self.dceval.SetAttValue("FILE", True)
        self.deval.SetAttValue("FILE", True)
        self.tteval.SetAttValue("FILE", True)
        # self.linkeval.SetAttValue("FILE", True)
        # 信号控制机、信号灯组
        self.sc, self.sg = [], []
        # 检测器、检测器保存数据
        self.tt, self.travel_time = [], []
        self.dt, self.delay = [], []
        self.dc, self.vel, self.speed = [], [], []
        self.qc, self.queue_length = [], []
        # 车流输入
        self.ip, self.ip_flow = [], []
        # 信号周期、相位差（周期延迟时间）、运行时间
        self.ct = 0
        self.offset = 0
        self.elapsed_time = 0

    def load_net(self, file):
        """load vissim road net file of *.inp format"""
        self.net_file = file.replace("/", "\\")
        self.vissim.LoadNet(self.net_file)

    def simulation_setting(self, period=999999, speed=0, resolution=1, controllerfrequency=1, randomseed=42):
        """"simulation software running parameter setting"""
        self.simulation.SetAttValue("PERIOD", period)
        self.simulation.Speed = speed
        self.simulation.Resolution = resolution
        self.simulation.ControllerFrequency = controllerfrequency
        self.simulation.RandomSeed = randomseed
        if self.simulation.RandomSeed == 0:
            self.simulation.RandomSeed = randint(1, 9999)
        self.simulation_stop_time = period

    def graphics_setting(self, visualization=True, mode_3d=True):
        """graphical interface parameter setting"""
        self.graphics.SetAttValue("VISUALIZATION", visualization)
        self.graphics.SetAttValue("3D", mode_3d)

    """Vissim运行控制API"""

    def start(self):
        """start of vissim simulation"""
        if self.vissim is None:
            raise ValueError("Could not find VISSIM COM server, please run VISSIM or register VISSIM_COMServer\n")
        self.connect()
        # 设置检测器
        self.set_signal_controller()
        self.set_signal_group()
        self.set_data_collections_detector()
        self.set_travel_times_detector()
        self.set_delay_times_detector()
        self.set_queue_counters_detector()

    def run_single_step(self):
        """run vissim simulation of single step"""
        self.simulation.RunSingleStep()

    def run_continuous(self):
        """run vissim simulation of continue"""
        self.simulation.RunContinuous()

    def stop(self):
        """stop of vissim simulation"""
        if self.vissim is not None:
            self.simulation.Stop()

    def end(self):
        """end of vissim simulation"""
        if self.vissim is not None:
            # 正在运行中，先停止仿真，再退出程序
            if self.simulation.AttValue("ELAPSEDTIME") > 0:
                self.stop()
                self.vissim.Exit()
                self.vissim = None
            # 仿真没有运行，直接退出程序
            else:
                self.vissim.Exit()
                self.vissim = None

    @staticmethod
    def detect_vissim():
        """detect whether vissim is running through background process name"""
        pids = psutil.pids()
        for pid in pids:
            p = psutil.Process(pid)
            process_name = p.name()
            if "vissim" in process_name:
                return True
        else:
            return False

    """交通流量设置API"""

    # 设置仿真输入车流量
    def set_vehicle_input_flow(self, flow):
        self.ip, self.ip_flow = [], []
        for i in range(self.inputs.Count):
            vehin = self.inputs.GetVehicleInputByNumber(i + 1)
            vehin.SetAttValue('VOLUME', flow[i])
            self.ip.append(vehin)
            self.ip_flow.append(flow[i])

    """信号灯设置API"""

    # 设置信号控制机(控制机<->交叉口)
    def set_signal_controller(self):
        self.sc = []
        for i in range(self.controllers.Count):
            controller = self.controllers.GetSignalControllerByNumber(i + 1)
            self.sc.append(controller)

    # 设置信号灯组(灯组<->相位)
    def set_signal_group(self):
        self.sg = []
        for i in range(self.groups.Count):
            group = self.groups.GetSignalGroupByNumber(i + 1)
            self.sg.append(group)

    """信号灯控制API"""

    # 控制信号灯组
    def control_signal_group(self, amber_time, clear_time, green_time):
        """
        Args:
           amber_time: The yellow light time of each phase
           clear_time: Red light clearance time of each phase
           green_time: Green time of each phase
           An example is as follows: [3, 3, 3], [2, 2, 2], [43, 55, 57]
        Returns: No return value
        """
        phase_num = len(green_time)
        value = [0 for _ in range(phase_num * 2)]
        for i in range(phase_num):
            if i == 0:
                value[i * 2] = 1
            else:
                value[i * 2] = value[i * 2 - 1] + amber_time[i] + clear_time[i - 1]
            value[i * 2 + 1] = value[i * 2] + green_time[i]
            self.sg[i].SetAttValue("REDEND", value[i * 2])
            self.sg[i].SetAttValue("GREENEND", value[i * 2 + 1])

    """检测器设置API"""

    # 设置行程时间检测器（检测器<->交叉口进出口一条完整道路的起点到终点之间的区域，需要在visism路网中设置检测器）
    def set_travel_times_detector(self):
        self.tt, self.travel_time = [], []
        for i in range(self.travel_times.Count):
            travel_time = self.travel_times.GetTravelTimeByNumber(i + 1)
            self.tt.append(travel_time)
            self.travel_time.append(0)  # 检测器行程时间初值

    # 设置延误时间检测器（检测器<->整个交叉口平均延误，即交叉口内多条道路的平均延误，需要在visism路网中设置检测器）
    def set_delay_times_detector(self):
        self.dt, self.delay = [], []
        for i in range(self.delays.Count):
            delay = self.delays.GetDelayByNumber(i + 1)
            self.dt.append(delay)
            self.delay.append(0)  # 检测器延误时间初值

    # 设置数据采集检测器（检测器<->进口道停车线前矩形区域内，需要在visism路网中设置检测器）
    def set_data_collections_detector(self):
        self.dc, self.vel, self.speed = [], [], []
        for i in range(self.data_collections.Count):
            data_collection = self.data_collections.GetDataCollectionByNumber(i + 1)
            self.dc.append(data_collection)
            self.vel.append(0)  # 检测器车流量初值
            self.speed.append(0)  # 检测器车速初值

    # 设置排队长度检测器（检测器<->进口道停车线，需要在visism路网中设置检测器）
    def set_queue_counters_detector(self):
        self.qc, self.queue_length = [], []
        for i in range(self.queue_counters.Count):
            queue_counter = self.queue_counters.GetQueueCounterByNumber(i + 1)
            self.qc.append(queue_counter)
            self.queue_length.append(0)  # 检测器排队长度初值

    """检测器数据提取API"""

    # 获取车流量采集检测器数据（需要在vissin软件的Evaluation菜单下的Files选项中设置）
    def get_flow_collections_detector(self, parameter="NVEHICLES", function="sum", vehicleclass=0):
        """
        [in]parameter:
            1、ACCELERATION: Acceleration [m/s2] [ft/s2]. MIN, MAX, MEAN, FREQUENCIES
            2、LENGTH: Vehicle length [m] [ft]. MIN, MAX, MEAN, FREQUENCIES
            3、MOTOTEMPERATURE: Cooling water temperature [°C]. MIN, MAX, MEAN, FREQUENCIES
            4、NVEHICLES: Number of vehicles. SUM
            5、NPERSONS: Number of people. MIN, MAX, MEAN, SUM, FREQUENCIES
            6、OCCUPANCYRATE: Occupancy rate [%]. SUM
            7、QUEUEDELTIME: Total queue delay time [s]. MIN, MAX, MEAN, SUM, FREQUENCIES
            8、SPEED: Speed [km/h] [mph]. MIN, MAX, MEAN, FREQUENCIES
            9、TACHO: Total distance traveled in the network [m] [ft]. MIN, MAX, MEAN, FREQUENCIES
        [in]function:
            1、min: Minimum value
            2、max: Maximum value
            3、mean: Mean value
            4、sum: Total sum
            4、frequencies: All configured frequencies in an array
        [in]vehicleclass: vehicle class number. 0 for all vehicle types
            1、0: all vehicle types
            2、1: vehicle for class 1(Car)
            3、2: vehicle for class 2(HGV)
            4、3: vehicle for class 3(Bus)
            5、4: vehicle for class 4(Tram)
            6、5: vehicle for class 5(Pedestrian)
            7、6: vehicle for class 6(Bike)
        [out]value: returned value (real) or array of values (of reals)
        Note: To receive results from collected data, a configuration must be defined in VISSIM and stored in
              a *.qmk file before requests can be done. The Offline Analysis option for data collections must be
              also enabled. Otherwise an error message will be returned (“The specified configuration is not defined
              within VISSIM”).
        """
        self.vel = []
        for i in range(len(self.dc)):
            self.vel.append(self.dc[i].GetResult(parameter, function, vehicleclass))
        return self.vel

    # 获取平均车速采集检测器数据（需要在vissin软件的Evaluation菜单下的Files选项中设置）
    def get_speed_collections_detector(self, parameter="SPEED", function="mean", vehicleclass=0):
        """
        [in]parameter:
            1、ACCELERATION: Acceleration [m/s2] [ft/s2]. MIN, MAX, MEAN, FREQUENCIES
            2、LENGTH: Vehicle length [m] [ft]. MIN, MAX, MEAN, FREQUENCIES
            3、MOTOTEMPERATURE: Cooling water temperature [°C]. MIN, MAX, MEAN, FREQUENCIES
            4、NVEHICLES: Number of vehicles. SUM
            5、NPERSONS: Number of people. MIN, MAX, MEAN, SUM, FREQUENCIES
            6、OCCUPANCYRATE: Occupancy rate [%]. SUM
            7、QUEUEDELTIME: Total queue delay time [s]. MIN, MAX, MEAN, SUM, FREQUENCIES
            8、SPEED: Speed [km/h] [mph]. MIN, MAX, MEAN, FREQUENCIES
            9、TACHO: Total distance traveled in the network [m] [ft]. MIN, MAX, MEAN, FREQUENCIES
        [in]function:
            1、min: Minimum value
            2、max: Maximum value
            3、mean: Mean value
            4、sum: Total sum
            4、frequencies: All configured frequencies in an array
        [in]vehicleclass: vehicle class number. 0 for all vehicle types
            1、0: all vehicle types
            2、1: vehicle for class 1(Car)
            3、2: vehicle for class 2(HGV)
            4、3: vehicle for class 3(Bus)
            5、4: vehicle for class 4(Tram)
            6、5: vehicle for class 5(Pedestrian)
            7、6: vehicle for class 6(Bike)
        [out]value: returned value (real) or array of values (of reals)
        Note: To receive results from collected data, a configuration must be defined in VISSIM and stored in
              a *.qmk file before requests can be done. The Offline Analysis option for data collections must be
              also enabled. Otherwise an error message will be returned (“The specified configuration is not defined
              within VISSIM”).
        """
        self.speed = []
        for i in range(len(self.dc)):
            self.speed.append(self.dc[i].GetResult(parameter, function, vehicleclass))
        return self.speed

    # 获取排队长度检测器数据（需要在vissin软件的Evaluation菜单下的Files选项中设置）
    def get_queue_counters_detector(self, elapsed_time=0, parameter="mean"):
        """
        [in]elapsed_time: Time point in seconds
        [in]parameter:
            1、MEAN: Average queue length ([m] or [ft], depending on current unit selection)
            2、MAX: Maximum queue length ([m] or [ft], depending on current unit selection)
            3、NSTOPS: Number of stops in the queue area
        [out]value: returned value (real number)
        Note: To get results, the Offline Analysis option for queue counters must be enabled.
              Otherwise the result will be 0.0.
        """
        if elapsed_time == 0:
            elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
        self.queue_length = []
        for i in range(len(self.qc)):
            self.queue_length.append(self.qc[i].GetResult(elapsed_time, parameter))
        return self.queue_length

    # 获取延误检测器数据（需要在vissin软件的Evaluation菜单下的Files选项中设置）
    def get_delay_times_detector(self, elapsed_time=0, parameter="DELAY", function="", vehicleclass=0):
        """
        [in]elapsed_time: Time point in seconds
        [in]parameter:
            1、DELAY: Average total delay per vehicle [s]
            2、PERSONS: Average total delay per person [s]
            3、NPERSONS: Persons throughput
            4、NVEHICLES: Vehicles throughput
            5、NSTOPS: Average number of stops per vehicle [s]
            6、STOPPEDDELAY: Average stand still time per vehicle [s]
        [in]function: not used
        [in]vehicleclass: vehicle class number. 0 for all vehicle types
            1、0: all vehicle types
            2、1: vehicle for class 1(Car)
            3、2: vehicle for class 2(HGV)
            4、3: vehicle for class 3(Bus)
            5、4: vehicle for class 4(Tram)
            6、5: vehicle for class 5(Pedestrian)
            7、6: vehicle for class 6(Bike)
        [out]value: returned value (real number)
        Note: To get results, the Offline Analysis option for delays must be enabled.
              Otherwise the result will be 0.0.
        """
        if elapsed_time == 0:
            elapsed_time = self.simulation.AttValue("ELAPSEDTIME")
        self.delay = []
        for i in range(len(self.dt)):
            self.delay.append(self.dt[i].GetResult(elapsed_time, parameter, function, vehicleclass))
        return self.delay
