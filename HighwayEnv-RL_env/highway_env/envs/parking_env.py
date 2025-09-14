from __future__ import annotations

from abc import abstractmethod

import numpy as np
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.
    目标条件连续控制任务，其中自我车辆必须以适当的航向停放在给定的空间中
    
        Goal-based环境：继承GoalEnv接口，支持目标导向学习
        连续控制：精确的转向和加速度控制
        停车场景：14个停车位的停车场
        目标停车：车辆必须停到指定位置且朝向正确
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
                "success_goal_reward": 0.12,
                "collision_reward": -5,
                "steering_range": np.deg2rad(45),
                "simulation_frequency": 15,   # 每秒进行15次物理仿真更新
                "policy_frequency": 5,        # 每秒进行5次动作决策
                "duration": 100,              # 持续时间
                "screen_width": 600,
                "screen_height": 300,
                "centering_position": [0.5, 0.5],   # 屏幕中心位置
                "scaling": 7,   # 视觉显示的缩放比例，用于控制停车环境在屏幕上的显示大小
                "controlled_vehicles": 1,
                "vehicles_count": 0,
                "add_walls": True,
            }
        )
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [x, y_offset], [x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [x, -y_offset], [x, -y_offset - length], width=width, line_types=lt
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            x0 = (i - self.config["controlled_vehicles"] // 2) * 10
            vehicle = self.action_type.vehicle_class(
                self.road, [x0, 0], 2 * np.pi * self.np_random.uniform(), 0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal
        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, 4, speed=0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def _reward(self, action: np.ndarray) -> float:
        """
        优化的停车奖励函数，提供更多正向激励和渐进式学习
        """
        import torch
        
        # 获取观测数据
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        
        total_reward = 0.0
        reward_components = {}
        
        for agent_obs in obs:
            achieved_goal = torch.tensor(agent_obs["achieved_goal"], dtype=torch.float32)
            desired_goal = torch.tensor(agent_obs["desired_goal"], dtype=torch.float32)
            
            # 1. 基础距离奖励（改进版）
            distance_temp: float = 1.5
            goal_distance = torch.norm(achieved_goal[:2] - desired_goal[:2])  # 位置距离
            heading_distance = torch.norm(achieved_goal[4:6] - desired_goal[4:6])  # 朝向距离
            
            # 使用更平滑的距离奖励
            distance_reward = torch.exp(-goal_distance * distance_temp)
            heading_reward = torch.exp(-heading_distance * distance_temp)
            
            # 2. 速度控制奖励（新增）
            speed_temp: float = 2.0
            current_speed = torch.norm(achieved_goal[2:4])  # 当前速度
            optimal_speed = torch.clamp(goal_distance * 0.5, 0.0, 5.0)  # 根据距离调整最优速度
            speed_diff = torch.abs(current_speed - optimal_speed)
            speed_reward = torch.exp(-speed_diff * speed_temp)
            
            # 3. 渐进式接近奖励
            approach_temp: float = 3.0
            if goal_distance < 10.0:  # 接近目标区域
                approach_bonus = torch.exp(-goal_distance * approach_temp)
            else:
                approach_bonus = torch.tensor(0.0)
            
            # 4. 稳定性奖励（低速接近时）
            stability_temp: float = 2.5
            if goal_distance < 5.0 and current_speed < 2.0:
                stability_bonus = torch.exp(-current_speed * stability_temp)
            else:
                stability_bonus = torch.tensor(0.0)
            
            # 5. 成功奖励增强
            success_threshold = self.config["success_goal_reward"]
            base_goal_reward = self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            
            if base_goal_reward > -success_threshold:
                success_multiplier = 3.0  # 成功时的奖励倍数
                enhanced_success_reward = base_goal_reward * success_multiplier
            else:
                enhanced_success_reward = base_goal_reward
            
            # 组合所有奖励组件
            component_reward = (
                float(distance_reward) * 0.3 +
                float(heading_reward) * 0.2 +
                float(speed_reward) * 0.25 +
                float(approach_bonus) * 0.15 +
                float(stability_bonus) * 0.1 +
                enhanced_success_reward * 0.4  # 主要奖励仍来自目标距离
            )
            
            total_reward += component_reward
            
            # 记录奖励组件
            reward_components.update({
                "distance_reward": float(distance_reward),
                "heading_reward": float(heading_reward),
                "speed_reward": float(speed_reward),
                "approach_bonus": float(approach_bonus),
                "stability_bonus": float(stability_bonus),
                "goal_reward": enhanced_success_reward
            })
        
        # 6. 碰撞惩罚（减轻但保持）
        collision_penalty = self.config["collision_reward"] * 0.5  # 减轻碰撞惩罚强度
        collision_count = sum(v.crashed for v in self.controlled_vehicles)
        collision_reward = collision_penalty * collision_count
        
        total_reward += collision_reward
        reward_components["collision_reward"] = collision_reward
        
        # 7. 存活奖励（新增）
        survival_bonus = 0.02  # 每步小幅正奖励
        total_reward += survival_bonus
        reward_components["survival_bonus"] = survival_bonus
        
        # 8. 动作平滑性奖励（减少抖动）
        action_smoothness_temp: float = 1.0
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < 0.5:  # 奖励小幅动作
            smoothness_reward = np.exp(-action_magnitude * action_smoothness_temp) * 0.05
        else:
            smoothness_reward = 0.0
        
        total_reward += smoothness_reward
        reward_components["smoothness_reward"] = smoothness_reward
        
        return total_reward
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.3,  # 降低p值，使奖励更平缓
    ) -> float:
        """
        改进的目标奖励计算，更平缓的惩罚曲线
        """
        # 使用更平缓的惩罚函数
        weighted_distance = np.dot(
            np.abs(achieved_goal - desired_goal),
            np.array(self.config["reward_weights"]),
        )
        
        # 添加基础奖励避免过于负面
        base_reward = 1.0
        distance_penalty = np.power(weighted_distance, p)
        
        return base_reward - distance_penalty

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    """
    继承基础环境：所有配置与ParkingEnv相同
    更频繁控制：policy_frequency: 1（每步都决策）
    更短时限：duration: 20（20秒完成停车）
    适用场景：需要更精细控制的情况
    """
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})

    def _reward(self, action: np.ndarray) -> float:
        """
        优化的ParkingEnvActionRepeat奖励函数：减少碰撞，提高总体奖励
        """
        import torch
        
        # 获取观测数据
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        
        total_reward = 0.0
        reward_components = {}
        
        for agent_obs in obs:
            achieved_goal = torch.tensor(agent_obs["achieved_goal"], dtype=torch.float32)
            desired_goal = torch.tensor(agent_obs["desired_goal"], dtype=torch.float32)
            
            # 1. 改进的距离奖励 - 更平滑的奖励曲线
            distance_temp: float = 1.2  # 降低温度，使奖励更平缓
            goal_distance = torch.norm(achieved_goal[:2] - desired_goal[:2])
            heading_distance = torch.norm(achieved_goal[4:6] - desired_goal[4:6])
            
            # 使用更宽松的距离奖励，避免过度惩罚
            distance_reward = torch.exp(-goal_distance * distance_temp) + 0.3  # 添加基础奖励
            heading_reward = torch.exp(-heading_distance * distance_temp) + 0.2
            
            # 2. 动作平滑性奖励 - 强烈鼓励温和的动作
            smoothness_temp: float = 0.8
            action_magnitude = torch.tensor(np.linalg.norm(action), dtype=torch.float32)
            # 大幅奖励小动作，严厉惩罚大动作
            if action_magnitude < 0.3:
                smoothness_reward = torch.exp(-action_magnitude * smoothness_temp) * 0.4
            elif action_magnitude < 0.6:
                smoothness_reward = torch.exp(-action_magnitude * smoothness_temp) * 0.2
            else:
                smoothness_reward = -action_magnitude * 0.3  # 惩罚过大动作
            
            # 3. 速度控制奖励 - 鼓励适度速度
            speed_temp: float = 1.5
            current_speed = torch.norm(achieved_goal[2:4])
            
            # 根据距离目标调整理想速度
            if goal_distance > 10.0:
                ideal_speed = 3.0  # 远离目标时可以快一些
            elif goal_distance > 5.0:
                ideal_speed = 2.0  # 中等距离
            else:
                ideal_speed = 1.0  # 接近目标时要慢
            
            speed_diff = torch.abs(current_speed - ideal_speed)
            speed_reward = torch.exp(-speed_diff * speed_temp) * 0.3
            
            # 4. 渐进式成功奖励
            progress_temp: float = 2.0
            if goal_distance < 15.0:  # 进入有效范围
                progress_bonus = torch.exp(-goal_distance * progress_temp) * 0.5
            else:
                progress_bonus = torch.tensor(0.1)  # 基础探索奖励
            
            # 5. 稳定性奖励 - 奖励在目标附近的稳定行为
            stability_temp: float = 2.0
            if goal_distance < 8.0 and current_speed < 1.5:
                stability_bonus = torch.exp(-current_speed * stability_temp) * 0.6
            else:
                stability_bonus = torch.tensor(0.0)
            
            # 6. 基础目标奖励 - 使用原始compute_reward但增强
            base_goal_reward = self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            success_threshold = self.config["success_goal_reward"]
            
            # 成功时给予大幅奖励提升
            if base_goal_reward > -success_threshold:
                enhanced_goal_reward = base_goal_reward * 2.0 + 1.0  # 成功奖励
            else:
                enhanced_goal_reward = base_goal_reward * 0.8 + 0.5  # 减轻失败惩罚
            
            # 组合所有奖励组件
            component_reward = (
                float(distance_reward) * 0.25 +
                float(heading_reward) * 0.20 +
                float(speed_reward) * 0.25 +
                float(smoothness_reward) * 0.35 +  # 提高平滑性权重
                float(progress_bonus) * 0.15 +
                float(stability_bonus) * 0.10 +
                enhanced_goal_reward * 0.3
            )
            
            total_reward += component_reward
            
            # 记录奖励组件
            reward_components.update({
                "distance_reward": float(distance_reward),
                "heading_reward": float(heading_reward),
                "speed_reward": float(speed_reward),
                "smoothness_reward": float(smoothness_reward),
                "progress_bonus": float(progress_bonus),
                "stability_bonus": float(stability_bonus),
                "goal_reward": enhanced_goal_reward
            })
        
        # 7. 碰撞惩罚 - 适度减轻但保持威慑
        collision_penalty = self.config["collision_reward"] * 0.6  # 从-5减少到-3
        collision_count = sum(v.crashed for v in self.controlled_vehicles)
        collision_reward = collision_penalty * collision_count
        
        total_reward += collision_reward
        reward_components["collision_reward"] = collision_reward
        
        # 8. 存活奖励 - 大幅提高基础奖励
        survival_bonus = 0.08  # 从0.02增加到0.08
        total_reward += survival_bonus
        reward_components["survival_bonus"] = survival_bonus
        
        # 9. 时间效率奖励 - 奖励快速完成
        time_efficiency_temp: float = 1.0
        time_ratio = self.time / self.config["duration"]
        if hasattr(self, '_episode_success') and self._episode_success:
            # 如果成功，奖励快速完成
            efficiency_reward = torch.exp(-time_ratio * time_efficiency_temp) * 0.2
        else:
            # 鼓励持续尝试
            efficiency_reward = (1.0 - time_ratio) * 0.1
        
        total_reward += float(efficiency_reward)
        reward_components["efficiency_reward"] = float(efficiency_reward)
        
        return total_reward
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.25,  # 进一步降低p值
    ) -> float:
        """
        更宽松的目标奖励计算
        """
        # 计算加权距离
        weighted_distance = np.dot(
            np.abs(achieved_goal - desired_goal),
            np.array(self.config["reward_weights"]),
        )
        
        # 更宽松的奖励函数
        base_reward = 1.5  # 提高基础奖励
        distance_penalty = np.power(weighted_distance, p)
        
        # 添加距离阈值奖励
        if weighted_distance < 2.0:  # 很接近目标
            proximity_bonus = 0.8
        elif weighted_distance < 5.0:  # 较接近目标
            proximity_bonus = 0.4
        else:
            proximity_bonus = 0.0
        
        return base_reward - distance_penalty + proximity_bonus


class ParkingEnvParkedVehicles(ParkingEnv):
    """
    增加障碍：停车场中有10辆已停车的车辆
    更高难度：需要在拥挤环境中找到停车位
    避障要求：必须避免与已停车辆碰撞
    现实模拟：更接近真实停车场景
    """
    def __init__(self):
        super().__init__({"vehicles_count": 10})
