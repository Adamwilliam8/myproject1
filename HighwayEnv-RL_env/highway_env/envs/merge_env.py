from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):
    """
    自主车辆从主干道出发，但很快就到达了一个路口，入口匝道上正有车辆驶入。
    此时，智能体的目标是保持高速行驶，同时为车辆腾出空间，以便它们能够安全地汇入车流

        主要挑战：在高速公路入口处处理车辆合流
        双重目标：保持自身高速行驶 + 为其他车辆让出合流空间
        社会性驾驶：需要考虑其他车辆的需求（利他主义奖励）
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
                "collision_reward": -3.0,      # 加强碰撞惩罚
                "right_lane_reward": 0.1,     # 适度降低
                "high_speed_reward": 0.6,      # 大幅提高
                "lane_change_reward": -0.01,   # 降低变道惩罚
                "merging_speed_reward": -0.05, # 大幅降低利他主义惩罚
                "safety_bonus": 0.4,           # 新增安全奖励
                "progress_reward": 0.3,       # 新增进展奖励
                "reward_speed_range": [15, 25]  # 速度范围
            })
        return cfg

    def _reward(self, action: int) -> float:
        """
        全新设计的MergeEnv奖励函数，根本性解决碰撞和学习问题
        """
        rewards = self._rewards(action)
        
        # 计算加权总奖励
        reward = sum(
            self.config.get(name, 0) * reward_value
            for name, reward_value in rewards.items()
        )
        
        # 归一化到[0,1]范围
        return utils.lmap(
            reward,
            [
                self.config["collision_reward"] + self.config["merging_speed_reward"],
                self.config["high_speed_reward"] + self.config["right_lane_reward"] + 
                self.config["safety_bonus"] + self.config["progress_reward"]
            ],
            [0, 1],
        )
    
    def _rewards(self, action: int) -> dict[str, float]:
        """
        彻底重新设计的奖励分解，优先解决安全和速度问题
        """
        import torch
        import numpy as np
        
        # 基础速度标准化
        scaled_speed = utils.lmap(
            self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        
        # 1. 强化碰撞惩罚（立即终止恶性行为）
        collision_reward = -1.0 if self.vehicle.crashed else 0.0
        
        # 2. 激进的高速奖励（解决速度过低问题）
        speed_temp: float = 4.0
        speed_boost = 0.5  # 大幅基础提升
        enhanced_speed = max(0.0, scaled_speed) + speed_boost
        speed_tensor = torch.tensor(enhanced_speed * speed_temp)
        speed_reward = torch.exp(speed_tensor) - 1.0
        max_exp = torch.exp(torch.tensor((1.0 + speed_boost) * speed_temp))
        speed_reward = speed_reward / (max_exp - 1.0)
        
        # 3. 简化右车道奖励（减少复杂性）
        right_lane_temp: float = 1.8
        lane_id = float(self.vehicle.lane_index[2])
        lane_ratio = lane_id  # 车道1比车道0获得更高奖励
        lane_tensor = torch.tensor(lane_ratio * right_lane_temp)
        right_lane_reward = torch.exp(lane_tensor) - 1.0
        right_lane_reward = right_lane_reward / (torch.exp(torch.tensor(right_lane_temp)) - 1.0)
        
        # 4. 革命性安全奖励（防碰撞核心）
        safety_bonus = self._calculate_revolutionary_safety_bonus()
        
        # 5. 大幅简化利他主义（降低学习难度）
        merging_reward = self._calculate_simplified_merging_reward()
        
        # 6. 强化进展奖励（鼓励完成任务）
        progress_reward = self._calculate_boosted_progress_reward()
        
        # 7. 简化变道奖励
        lane_change_reward = -0.05 if action in [0, 2] else 0.0
        
        return {
            "collision_reward": collision_reward,
            "right_lane_reward": float(torch.clamp(right_lane_reward, 0.0, 1.0)),
            "high_speed_reward": float(torch.clamp(speed_reward, 0.0, 1.0)),
            "lane_change_reward": lane_change_reward,
            "merging_speed_reward": merging_reward,
            "safety_bonus": safety_bonus,
            "progress_reward": progress_reward,
        }
    
    def _calculate_revolutionary_safety_bonus(self) -> float:
        """
        革命性安全奖励：激进的安全优先策略
        """
        safety_temp: float = 3.0
        
        # 获取最关键的前车信息
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self.vehicle)
        
        # 激进的基础安全奖励
        safety_score = 0.8  # 高基础安全分
        
        if front_vehicle is not None:
            distance = np.linalg.norm(front_vehicle.position - self.vehicle.position)
            relative_speed = self.vehicle.speed - front_vehicle.speed
            
            # 更宽松的安全距离计算（降低难度）
            safe_distance = 5.0 + 0.2 * self.vehicle.speed + max(0, relative_speed) * 0.8
            safety_ratio = min(distance / safe_distance, 3.0)  # 更高上限
            
            if distance < 10.0:  # 危险距离内
                safety_score += safety_ratio * 0.5
            else:  # 安全距离内
                safety_score += 1.0  # 满分安全奖励
        else:
            safety_score += 1.2  # 无前车时的高奖励
        
        # 后车简化处理
        if rear_vehicle is not None:
            distance = np.linalg.norm(rear_vehicle.position - self.vehicle.position)
            if distance > 8.0:  # 安全距离
                safety_score += 0.5
        else:
            safety_score += 0.5
        
        # 指数函数放大安全奖励
        safety_tensor = torch.tensor(min(safety_score, 3.5) * safety_temp)
        safety_bonus = torch.exp(safety_tensor) - 1.0
        max_safety_exp = torch.exp(torch.tensor(3.5 * safety_temp))
        safety_bonus = safety_bonus / (max_safety_exp - 1.0)
        
        return float(torch.clamp(safety_bonus, 0.5, 1.0))  # 保证最低0.5安全奖励
    
    def _calculate_simplified_merging_reward(self) -> float:
        """
        大幅简化的合流奖励：降低学习复杂度
        """
        # 仅在合流区域且真正需要时给予轻微惩罚
        x_position = self.vehicle.position[0]
        
        if not (150 <= x_position <= 310):  # 不在合流区域
            return 0.0
        
        # 查找合流车辆
        merging_vehicles = [
            vehicle for vehicle in self.road.vehicles
            if (hasattr(vehicle, 'lane_index') and 
                isinstance(vehicle, self.action_type.vehicle_class) and
                vehicle != self.vehicle)
        ]
        
        if not merging_vehicles:
            return 0.0
        
        # 非常轻微的利他主义惩罚
        total_penalty = 0.0
        for vehicle in merging_vehicles:
            distance = np.linalg.norm(vehicle.position - self.vehicle.position)
            if distance < 30.0:  # 只考虑附近车辆
                speed_diff = max(0, vehicle.target_speed - vehicle.speed) / max(1, vehicle.target_speed)
                total_penalty += speed_diff * 0.1  # 大幅降低惩罚
        
        return -min(total_penalty, 0.2)  # 限制最大惩罚为0.2
    
    def _calculate_boosted_progress_reward(self) -> float:
        """
        大幅提升的进展奖励
        """
        # 基于速度的基础进展
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        speed_progress = min(max(forward_speed, 0.0) / 20.0, 1.5)  # 降低速度要求
        
        # 位置进展奖励
        x_position = self.vehicle.position[0]
        
        # 不同阶段的进展奖励
        if x_position < 150:  # 合流前
            position_progress = 0.3
        elif 150 <= x_position <= 310:  # 合流区域
            merge_progress = (x_position - 150) / 160.0
            position_progress = 0.5 + merge_progress * 0.4
        else:  # 合流后
            position_progress = 1.0
        
        total_progress = speed_progress + position_progress
        return min(total_progress, 2.0)

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        merging_v = other_vehicles_type(
            road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20
        )
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
