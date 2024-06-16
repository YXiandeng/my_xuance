import csv
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
from copy import deepcopy


class New_Env_MAS:
    f_c = 2e9  # 载波频率2GHz
    sitaL = 1
    sitaNL = 20
    pt = 10 ** ((30 - 30) / 10)  # 30dBm = 1W       每个无人机的发射功率
    BW_total = 30e6  # 无人机系统的总带宽30MHZ
    bw_min = 1e5  # 最小的可分带宽 0.1MHz
    n_0 = 10 ** ((-170 - 30) / 10)  # 噪声的功率谱密度 -170dBm/hz
    uav_h = 500  # 无人机的高度 500m
    v_uav = 15  # 无人机的飞行速度15m/s
    uav_range = 50  # 无人机的通信半径
    A = sitaL - sitaNL
    a = 9.61
    b = 0.28

    """
    Ground Users(GU) definition
    """
    num_users = 200  # 埋压人员数量
    raw = []
    with open('settings/u_loc2.txt', 'r') as f:
        for line in f:
            raw.append(line.split())
    all_users_loc = np.array(raw).astype(float)

    def __init__(self, env_id: str, seed: int, **kwargs):
        """
        Space Borders
        """
        # 任务区域 0-1000m
        self.MIN_X = 0
        self.MAX_X = 1000
        self.MIN_Y = 0
        self.MAX_Y = 1000

        self.borders = np.array([np.array([self.MIN_X, self.MAX_X]), np.array([self.MIN_Y, self.MAX_Y])])  # 设置边界
        self.n_agents = 4
        self.dim_obs = 2 + self.num_users  # 一个智能体观察的维度
        self.dim_state = self.dim_obs * self.n_agents  # 全局状态的维度
        self.dim_action = self.num_users + self.num_users + 1  # 功率  + 带宽 +无人机飞行方位角  # 动作维度（连续）
        self.n_actions = 5  # 离散操作数（离散）

        """
        UAVs definition
        """
        self.all_uavs_location = []
        self.num_uavs = self.n_agents
        self.uavs_id = ['uav_' + str(i) for i in range(self.num_uavs)]
        for uav_id in self.uavs_id:
            uavs_location = self.create_uavs(uav_id)
            self.all_uavs_location.append(uavs_location)
        self.all_uavs_location = np.array(self.all_uavs_location)

        # self.state_space = Box(low=0, high=1, shape=[self.dim_state, ], dtype=np.float32, seed=seed)  # 整体的状态
        self.state_space = np.zeros((self.dim_state,), dtype=np.float32)

        self.observation_space = np.zeros((self.dim_obs,), dtype=np.float32)  # 单无人机获得状态

        if kwargs['continuous']:
            self.action_space = Box(low=-1, high=1, shape=[self.dim_action, ], dtype=np.float32, seed=seed)
            # self.action_space = np.zeros((self.dim_action,), dtype=np.float32)
        else:
            self.action_space = Discrete(n=self.n_actions, seed=seed)

        self._episode_step = 0
        self._episode_score = 0.0

        self.max_episode_steps = 1000  # 回合的最大步数
        self.env_info = {
            "n_agents": self.n_agents,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
            "episode_limit": self.max_episode_steps,
        }
        self.jain_f_m_ep_avg = []

    def close(self):
        pass

    def render(self):
        return np.zeros([2, 2, 2])

    def reset(self):
        uav_loc_init = np.array([[0, 0], [0, 1000], [1000, 0], [1000, 1000]])
        # uav_loc_init = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        uav_loc_i = []
        obs_n = []
        connect = np.zeros((self.num_uavs, self.num_users))
        for uavs_id in self.uavs_id:
            uav_loc_i.append(self.create_uavs(uavs_id))
        for i in range(self.num_uavs):
            obs = list(uav_loc_init[i]) + list(connect[i])
            obs = np.array(obs.copy())
            self.observation_space = obs
            obs_n.append(obs)
        obs = np.array(obs_n)
        self.state_space = obs.copy()
        info = {}
        self._episode_step = 0
        self._episode_score = 0.0
        info["episode_step"] = self._episode_step
        self.jain_f_m_ep_avg = []
        return self.state_space, info

    def step(self, actions):
        info = {}
        # action (功率、带宽、角度)
        observation_space = self.state_space.reshape((self.num_uavs, self.dim_obs))
        # 执行操作并获取下一个观察结果、奖励和其他信息
        all_distence = np.zeros((self.num_uavs, self.num_users))  # 无人机与所有的地面人员距离
        all_tmp = np.zeros((self.num_uavs, self.num_users))  # 无人机与地面人员的水平距离
        # 功率
        power_all = []
        # 带宽
        bandwidth_all = []
        # 连接状态
        connect_pr_all = []
        # 每个无人机的奖励
        rewards = []
        # 每个无人机的负载情况
        s_m_all = np.zeros((1, self.num_uavs))
        terminated = [False for _ in range(self.n_agents)]
        if self._episode_step >= self.max_episode_steps:
            terminated = [True for _ in range(self.n_agents)]

        # set action for each agent
        for uav_idx in range(self.num_uavs):
            # 状态分解(无人机的二维位置、每个地面人员与该无人机的连接情况)
            state_pr_i = observation_space[uav_idx]
            prev_location = state_pr_i[0:2].astype(float).copy()
            connect_pr_i = state_pr_i[2:].astype(float).copy()
            for i in range(len(connect_pr_i)):
                if connect_pr_i[i] > 0.5:
                    connect_pr_i[i] = 1
                else:
                    connect_pr_i[i] = 0
            connect_pr_all.append(connect_pr_i)
            # 动作掩码
            k_mask = connect_pr_i.astype(float)

            # 动作分解（功率、带宽、飞行角度）
            # 网络输出的动作需要编码解码成真正的功率和带宽
            action_n_i = actions[uav_idx]
            # 角度sita编码为实际角度[-pi,pi]
            sita = action_n_i[-1].copy() * np.pi

            # 分配的功率
            power_n_i = action_n_i[:200].copy()
            power_n_i = (power_n_i + 1) / 2
            power_n_i = power_n_i * k_mask
            s_m = len([x for x in k_mask if x != 0])
            s_m_all[0, uav_idx] = s_m
            for k in range(len(power_n_i)):
                if k_mask[k] == 0:
                    power_n_i[k] = 0
                else:
                    power_n_i[k] = self.pt * (np.exp(power_n_i[k]) / (np.sum(np.exp(power_n_i))))
            power_all.append(power_n_i)

            # 分配的带宽
            b_m = self.BW_total * (s_m / self.num_users)
            bandwidth_n_i = action_n_i[201:401].copy()
            bandwidth_n_i = bandwidth_n_i * k_mask
            bandwidth_n_i = (bandwidth_n_i + 1) / 2
            for m in range(len(bandwidth_n_i)):
                if k_mask[m] == 0 or s_m == 0:
                    bandwidth_n_i[m] = 0
                else:
                    bandwidth_n_i[m] = self.bw_min + ((np.exp(bandwidth_n_i[m]) / (np.sum(np.exp(bandwidth_n_i)))) * (
                            b_m - (s_m * self.bw_min)))
            bandwidth_all.append(bandwidth_n_i)

            # 飞行动作
            # 设置每个智能体的动作
            if sita > np.pi or sita < -np.pi:
                sita = 0
            velocity_uav = np.multiply(np.array([np.cos(sita), np.sin(sita)]), self.v_uav)
            uav_loc = np.add(prev_location, velocity_uav)
            if not self.valid_location(uav_loc):
                if uav_loc[0] < self.borders[0][0] or uav_loc[0] > self.borders[0][1]:
                    uav_loc = deepcopy(prev_location)
                if uav_loc[1] < self.borders[1][0] or uav_loc[1] > self.borders[1][1]:
                    uav_loc = deepcopy(prev_location)
            self.all_uavs_location[uav_idx, 0:2] = uav_loc

        # 删除第三列
        all_uavs_loc = np.array(np.delete(self.all_uavs_location, 2, axis=1))
        # 所有的带宽
        all_bandwidth = np.array(bandwidth_all)
        # 所有的连接状态
        connect_pr = np.array(connect_pr_all)
        # 所有的功率
        power_all = np.array(power_all)

        # 无人机与地面用户距离
        for j in range(self.num_uavs):
            for i in range(self.num_users):
                all_tmp[j][i] = np.linalg.norm(all_uavs_loc[j].astype(float) - self.all_users_loc[i].astype(float))
                all_distence[j][i] = math.sqrt(all_tmp[j][i] ** 2 + self.uav_h ** 2)
        result_array = np.zeros_like(all_distence)
        min_indices = np.argmin(all_distence, axis=0)
        result_array[min_indices, np.arange(all_distence.shape[1])] = 1
        # 连接状态
        all_connect_new = result_array

        obs_n = np.zeros((self.num_uavs, self.num_users + 2))
        obs_n[:, :2] = all_uavs_loc  # 将数组A放在数组B的前四行俩列中
        print(all_uavs_loc)
        # # 定义文件名
        # # filenames = ['uav_location_1.csv', 'uav_location_2.csv', 'uav_location_3.csv', 'uav_location_4.csv']
        # filenames = ['uav_location_matd3_1.csv', 'uav_location_matd3_2.csv', 'uav_location_matd3_3.csv', 'uav_location_matd3_4.csv']
        # # 将数组的每一行写入到对应的CSV文件中
        # for row, file_name in zip(all_uavs_loc, filenames):
        #     # 打开文件，以追加模式写入数据
        #     with open(file_name, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow(row)

        obs_n[:, -200:] = all_connect_new  # 将数组C放在数组B的后四行200列中
        for uav_idx in range(self.num_uavs):
            info["uav-station-%d-x" % uav_idx] = all_uavs_loc[uav_idx][0]
            info["uav-station-%d-y" % uav_idx] = all_uavs_loc[uav_idx][1]
        # record observation for each agent
        observation = obs_n
        self.state_space = obs_n.reshape(self.dim_state, )

        comm_rate = self.calc_comm_rate(all_uavs_loc, power_all, connect_pr, all_bandwidth)
        jain_f_m = 0
        for uav_idx in range(self.num_uavs):
            jain_f_m_i = self.jain_f_m_i(comm_rate[uav_idx], s_m_all[0, uav_idx])
            jain_f_m += jain_f_m_i
            rewards.append(self.reward_function_i(all_uavs_loc, comm_rate[uav_idx], uav_idx, s_m_all[0, uav_idx]))
        jain_f_m_avg = jain_f_m / self.num_uavs
        self.jain_f_m_ep_avg.append(jain_f_m_avg)
        reward_n = sum(rewards)
        # # 转化为Mbits
        reward_n = reward_n/1e6
        collision_reward = 0  # 碰撞惩罚
        reward_n = reward_n + collision_reward
        reward = np.zeros([self.n_agents, 1])
        for i in range(self.n_agents):
            reward[i] = reward_n
        truncated = [True for _ in range(self.n_agents)] if (self._episode_step >= self.max_episode_steps) else [False
                                                                                                                 for _
                                                                                                                 in
                                                                                                                 range(
                                                                                                                     self.n_agents)]

        self._episode_step += 1
        self._episode_score += reward
        info["episode_step"] = self._episode_step  # current episode step
        info["jain_f_m"] = self.jain_f_m_ep_avg
        info["episode_score"] = self._episode_score  # the accumulated rewards
        return observation, reward, terminated, truncated, info

    def get_agent_mask(self):
        return np.ones(self.n_agents, dtype=np.bool_)

    def state(self):
        return np.zeros([self.dim_state])

    def jain_f_m_i(self, comm_rate, s_m):
        # comm_rate 是一个数组
        # Jain公平指数系数 c: 与无人机m通信的GU的通信速率  s:与无人机m通通信的GU个数
        sum_c = np.sum(comm_rate)
        c_square_sum = np.sum(np.square(comm_rate))
        if c_square_sum == 0:
            f_m = 0
        else:
            f_m = sum_c ** 2 / (s_m * c_square_sum)
        return f_m

    def jain_f_m_avg(self, jain_f_m):
        jain_f_m_avg = np.sum(jain_f_m) / self.num_uavs
        return jain_f_m_avg

    def reward_function_i(self, all_uav_location, comm_rate, uavs_id, s_m):
        r_d = 2
        r_b = 200
        k_r = 5
        k_b = 8e-2
        reward_weights = np.array([1, 1])  # 各个部分的奖励权重
        if s_m == 0:
            com_rate_mean = 0
        else:
            com_rate_mean = (np.sum(comm_rate)) / s_m
        com_rate_reward = 0  # 通信奖励
        boundary_reward = 0  # 位置惩罚
        # 无人机的位置坐标
        # 通信奖励
        com_rate_reward = r_d * (self.jain_f_m_i(comm_rate, s_m) ** k_r) * com_rate_mean
        # 边界奖励
        boundary_reward = r_b * (1 / (1 + math.exp(
            -k_b * (np.square(all_uav_location[uavs_id][0].astype(float) ** 2 + all_uav_location[uavs_id][1].astype(
                float) ** 2)))) + 1 / (
                                         1 + math.exp(-k_b * (np.square(
                                     (all_uav_location[uavs_id][0].astype(float) - 1000) ** 2 + (
                                             all_uav_location[uavs_id][1].astype(float) - 1000) ** 2)))))
        reward = [com_rate_reward, -boundary_reward]  # 各个部分的奖励
        return np.matmul(reward, reward_weights)

    # 信干噪比
    def calc_comm_rate(self, all_uavs_loc, power, connect_pr, all_bandwidth):
        comm_rate = np.zeros((self.num_uavs, self.num_users))
        # 无人机与地面用户距离以及通信速率
        for j in range(self.num_uavs):
            for i in range(self.num_users):
                tmp = np.linalg.norm(all_uavs_loc[j].astype(float) - self.all_users_loc[j].astype(float))
                if connect_pr[j][i] == 0:
                    continue
                dis = math.sqrt(tmp ** 2 + self.uav_h ** 2)
                sita = np.arctan(self.uav_h / tmp)
                pl = self.A / (
                        1 + self.a * np.exp((-self.b) * ((180 * sita / math.pi) - self.a))) + 20 * np.log10(
                    dis) + 20 * np.log10(4 * math.pi * self.f_c / 3e8) + self.sitaNL

                g = 10 ** (-pl / 20)
                sinr = g * power[j][i] / (self.n_0 * all_bandwidth[j][i])
                comm_rate[j][i] = connect_pr[j][i] * all_bandwidth[j][i] * math.log(1 + sinr)
        return comm_rate

    def create_uavs(self, uav_id):
        uav_location = np.array([500, 500.], dtype=np.float64)
        uav_location = np.append(uav_location, uav_id)
        return uav_location

    def valid_location(self, location):
        if self.out_of_border(location):
            return False
        return True

    # 超出任务区域
    def out_of_border(self, location):
        if np.any(location < self.borders[0]) or np.any(location > self.borders[1]):
            return True
        return False
