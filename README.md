# visism-gym-environment

#### 介绍
该项目旨在打造一个交通仿真软件vissim的gym训练环境，将vissim的仿真操作封装为gym环境的5个api接口，包括reset、render、step、close、seed，基于此项目可以直接使用gym风格的强化学习算法控制交叉口信号灯，无需关注底层仿真交互细节，项目已经具备基本功能，欢迎各位有志之士加入项目

#### 软件架构
软件架构说明


#### 安装教程

1.  pip install gym 
2.  pip install  psutil
3.  pip install pypiwin32

#### 使用说明

1. 因本项目调用了vissim仿真软件，因此只能运行在windows操作系统上

2. vistraci文件夹是vissim的gym环境封装，首先通过 `pip install gym` 命令安装gym工具包

      然后复制整个vistraci文件夹到 `xx/Lib/site-packages/gym/envs`下

      最后在 `xx/Lib/site-packages/gym/envs/__init__.py` 文件中注册vissim环境

      注册方式为：在该\__init__.py文件最后加入以下代码

   ```python
   register(
     id='Vissim-v0',
     entry_point='gym.envs.vistraci:VissimEnv0',
     max_episode_steps=10**4,
     reward_threshold=100.0,
   )
   ```

3. RunAsDate文件夹下是启动vissim软件的相关文件，根据该文件夹下的`使用方法.txt`修改run vissim.bat文件，然后双击运行run vissim.bat即可启动visism

4. demo_random文件夹下存放一个随机选择配时动作的vissim仿真示例

3.  demo_dqn文件夹下存放一个根据DQN算法选择配时动作的vissim仿真示例

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
