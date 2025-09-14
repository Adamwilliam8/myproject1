from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle


class RoundaboutEnv(AbstractEnv):
    """
    在这个任务中，自主车辆正在接近一个车流流动的环岛。
    它会自动遵循规划好的路线，但必须处理车道变换和纵向控制，
    以便尽快通过环岛，同时避免碰撞
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
                "incoming_vehicle_destination": None,
                "collision_reward": -2,
                "high_speed_reward": 0.4,
                "right_lane_reward": 0,
                "lane_change_reward": -0.02,
                "progress_reward": 0.6,            # 高进展奖励权重
                "route_completion_reward": 1.0,     # 最高完成奖励权重
                "efficiency_reward": 0.3,          # 效率奖励权重
                "position_reward": 0.2,            # 位置奖励权重
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 15,
                "normalize_reward": True,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        优化的环岛奖励函数，显著提高任务完成能力和奖励值
        """
        rewards = self._rewards(action)
        
        # 加权求和计算总奖励
        reward = sum(
            self.config.get(name, 0) * reward_value
            for name, reward_value in rewards.items()
        )
        
        # 归一化到[0,1]范围
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"] + self.config["lane_change_reward"],
                    self.config["high_speed_reward"] + self.config.get("progress_reward", 0) + 
                    self.config.get("route_completion_reward", 0) + self.config.get("efficiency_reward", 0)
                ],
                [0, 1],
            )
        
        # 应用on_road乘数
        reward *= rewards["on_road_reward"]
        return reward
    
    def _rewards(self, action: int) -> dict[str, float]:
        """
        简化但完整的环岛奖励分解
        """
        import torch
        import numpy as np
        
        # 基础奖励组件
        collision_reward = float(self.vehicle.crashed)
        
        # 大幅增强的速度奖励
        speed_temp: float = 3.5
        speed_index = MDPVehicle.get_speed_index(self.vehicle)
        max_speed_index = len(self.config["action"]["target_speeds"]) - 1
        
        if max_speed_index > 0:
            speed_base = 0.3
            speed_ratio = float(speed_index) / float(max_speed_index) + speed_base
            speed_tensor = torch.tensor(speed_ratio * speed_temp)
            speed_reward = torch.exp(speed_tensor) - 1.0
            speed_reward = speed_reward / (torch.exp(torch.tensor((1.0 + speed_base) * speed_temp)) - 1.0)
        else:
            speed_reward = torch.tensor(0.0)
        
        # 简化的变道奖励
        if action in [0, 2]:  # 变道动作
            lane_change_reward = -0.05  # 轻微惩罚变道
        else:
            lane_change_reward = 0.0
        
        # 简化的进展奖励
        try:
            if hasattr(self.vehicle, 'route') and self.vehicle.route:
                current_lane_index = self.vehicle.lane_index
                route = self.vehicle.route
                if current_lane_index in route:
                    current_position = route.index(current_lane_index)
                    progress_reward = float(current_position) / float(len(route) - 1)
                else:
                    progress_reward = 0.0
            else:
                progress_reward = 0.0
        except:
            progress_reward = 0.0
        
        # 简化的路径完成奖励
        try:
            if hasattr(self.vehicle, 'route') and self.vehicle.route:
                current_lane_index = self.vehicle.lane_index
                target_lane_index = self.vehicle.route[-1]
                if (current_lane_index[0] == target_lane_index[0] and 
                    current_lane_index[1] == target_lane_index[1]):
                    route_completion_reward = 1.0
                else:
                    route_completion_reward = 0.0
            else:
                route_completion_reward = 0.0
        except:
            route_completion_reward = 0.0
        
        # 简化的效率奖励
        time_efficiency = max(0.0, (self.config["duration"] - self.time) / self.config["duration"])
        speed_efficiency = min(self.vehicle.speed / 16.0, 1.0)
        efficiency_reward = (time_efficiency * 0.6 + speed_efficiency * 0.4)
        
        # 简化的位置奖励
        try:
            if hasattr(self.vehicle, 'destination'):
                current_pos = self.vehicle.position
                destination = self.vehicle.destination
                distance = np.linalg.norm(destination - current_pos)
                position_reward = max(0.0, 1.0 - distance / 100.0)  # 距离越近奖励越高
            else:
                position_reward = 0.0
        except:
            position_reward = 0.0
        
        return {
            "collision_reward": -collision_reward,
            "high_speed_reward": float(torch.clamp(speed_reward, 0.0, 1.0)),
            "lane_change_reward": lane_change_reward,
            "progress_reward": progress_reward,
            "route_completion_reward": route_completion_reward,
            "efficiency_reward": efficiency_reward,
            "position_reward": position_reward,
            "on_road_reward": float(self.vehicle.on_road),
        }
    
    def _calculate_smart_lane_change_reward(self, action: int) -> float:
        """
        智能变道奖励：只在必要时变道，减少无效行为
        """
        if action not in [0, 2]:  # 非变道动作
            return 0.0
        
        # 检查变道是否有益
        current_lane_id = self.vehicle.lane_index[2]
        
        # 环岛内部，内圈（车道0）更高效
        if current_lane_id == 1 and action == 0:  # 从外圈变到内圈
            return 0.2  # 奖励有效变道
        elif current_lane_id == 0 and action == 2:  # 从内圈变到外圈
            # 只在接近出口时奖励
            if hasattr(self.vehicle, 'route') and self.vehicle.route:
                return 0.1  # 适度奖励准备出环岛
            else:
                return -0.1  # 惩罚无目的变道
        
        return -0.05  # 轻微惩罚其他变道
    
    def _calculate_route_progress_reward(self) -> float:
        """
        路径进展奖励：基于沿路径的进展程度
        """
        if not hasattr(self.vehicle, 'route') or not self.vehicle.route:
            return 0.0
        
        progress_temp: float = 2.5
        
        try:
            current_lane_index = self.vehicle.lane_index
            route = self.vehicle.route
            
            # 计算在路径中的进展
            if current_lane_index in route:
                current_position = route.index(current_lane_index)
                progress_ratio = float(current_position) / float(len(route) - 1)
            else:
                # 寻找最接近的路径段
                progress_ratio = 0.0
                for i, lane_idx in enumerate(route):
                    if (lane_idx[0] == current_lane_index[0] and 
                        lane_idx[1] == current_lane_index[1]):
                        progress_ratio = float(i) / float(len(route) - 1)
                        break
            
            # 使用指数函数放大进展奖励
            progress_tensor = torch.tensor(progress_ratio * progress_temp)
            progress_reward = torch.exp(progress_tensor) - 1.0
            progress_reward = progress_reward / (torch.exp(torch.tensor(progress_temp)) - 1.0)
            
            return float(torch.clamp(progress_reward, 0.0, 1.0))
            
        except:
            return 0.0
    
    def _calculate_route_completion_reward(self) -> float:
        """
        路径完成奖励：到达目标时给予大幅奖励
        """
        if not hasattr(self.vehicle, 'route') or not self.vehicle.route:
            return 0.0
        
        try:
            current_lane_index = self.vehicle.lane_index
            target_lane_index = self.vehicle.route[-1]  # 路径最后一段
            
            # 检查是否到达目标区域
            if (current_lane_index[0] == target_lane_index[0] and 
                current_lane_index[1] == target_lane_index[1]):
                return 1.5  # 大幅奖励完成路径
            
            # 检查是否接近目标
            if (current_lane_index[1] == target_lane_index[0]):
                return 0.8  # 接近目标的奖励
                
            return 0.0
        except:
            return 0.0
    
    def _calculate_efficiency_reward(self) -> float:
        """
        效率奖励：基于时间和速度的综合效率
        """
        # 时间效率：越早完成任务奖励越高
        time_efficiency = max(0.0, (self.config["duration"] - self.time) / self.config["duration"])
        
        # 速度效率：结合当前速度
        speed_efficiency = min(self.vehicle.speed / 16.0, 1.0)  # 16是最大目标速度
        
        # 综合效率奖励
        efficiency = (time_efficiency * 0.6 + speed_efficiency * 0.4) * 1.2
        
        return min(efficiency, 1.0)
    
    def _calculate_position_reward(self) -> float:
        """
        位置奖励：鼓励向目标方向移动
        """
        if not hasattr(self.vehicle, 'route') or not self.vehicle.route:
            return 0.0
        
        try:
            # 计算到目标的方向奖励
            current_pos = self.vehicle.position
            destination = self.vehicle.destination
            
            if np.linalg.norm(destination - current_pos) < 1.0:
                return 0.0
            
            # 计算朝向目标的程度
            direction_to_dest = (destination - current_pos) / np.linalg.norm(destination - current_pos)
            vehicle_direction = self.vehicle.direction
            
            # 计算方向一致性
            alignment = np.dot(direction_to_dest, vehicle_direction)
            position_reward = max(0.0, alignment) * 0.5
            
            return min(position_reward, 0.5)
        except:
            return 0.0

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        net.add_lane(
            "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2],
                [2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [-2 - a, -dev / 2 + delta_en],
                [-2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        net.add_lane(
            "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en, 2 + a],
                [dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        net.add_lane(
            "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a, dev / 2 - delta_en],
                [2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        net.add_lane(
            "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a],
                [-dev / 2 + delta_st, 2 + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en, -2 - a],
                [-dev / 2, -2 - a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125, 0),
            speed=8,
            heading=ego_lane.heading_at(140),
        )
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("we", "sx", 1),
            longitudinal=5 + self.np_random.normal() * position_deviation,
            speed=16 + self.np_random.normal() * speed_deviation,
        )

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("we", "sx", 0),
                longitudinal=20 * i + self.np_random.normal() * position_deviation,
                speed=16 + self.np_random.normal() * speed_deviation,
            )
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("eer", "ees", 0),
            longitudinal=50 + self.np_random.normal() * position_deviation,
            speed=16 + self.np_random.normal() * speed_deviation,
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
