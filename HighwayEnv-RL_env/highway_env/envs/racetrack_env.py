from __future__ import annotations

import numpy as np
import torch

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


class RacetrackEnv(AbstractEnv):
    """
    连续控制：使用连续动作空间
    复杂轨道：包含直道和多个弯道的复杂赛道布局
    双重任务：既要跟踪轨道又要避免碰撞
    观测类型：占用栅格（OccupancyGrid）
    动作配置：只有横向控制（转向），无纵向控制

    赛场：在连续控制环境中，代理必须沿着轨道行驶，同时避免与其他车辆发生碰撞
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
            }
        )
        return config
    
    def _reward(self, action: torch.Tensor) -> float:
        """
        优化的 RacetrackEnv 奖励函数
        """
        rewards = self._rewards(action)
        
        # 基于配置权重的加权总和
        reward = sum(
            self.config.get(name, 0) * reward_value 
            for name, reward_value in rewards.items()
        )

        # 线性映射到 [0, 1] 范围
        reward = utils.lmap(reward, 
                           [self.config["collision_reward"], 1], 
                           [0, 1])
        
        # 应用 on_road 乘数
        reward *= rewards["on_road_reward"]
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return float(reward)
    
    def _rewards(self, action: torch.Tensor) -> dict[str, float]:
        """
        多目标奖励组件分解
        """
        # 确保 action 是 torch.Tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
    
        # 获取车辆在车道中的横向位置
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
    
        # 转换为 tensor 并确保在同一设备上
        device = action.device if isinstance(action, torch.Tensor) else "cpu"
        longitudinal = torch.tensor(longitudinal, dtype=torch.float32, device=device)
        lateral = torch.tensor(lateral, dtype=torch.float32, device=device)
        current_speed = torch.tensor(self.vehicle.speed, dtype=torch.float32, device=device)
        target_speed = torch.tensor(self.config.get("speed_limit", 10.0), dtype=torch.float32, device=device)
        vehicle_heading = torch.tensor(self.vehicle.heading, dtype=torch.float32, device=device)
        
        # 温度参数
        lateral_temp: float = 2.0
        speed_temp: float = 1.5
        progress_temp: float = 1.0
        action_temp: float = 0.8
        heading_temp: float = 2.5
        
        # 1. 车道居中奖励 - 使用指数函数平滑奖励
        lateral_distance = torch.abs(lateral)
        lane_centering_reward = torch.exp(-lateral_distance * lateral_temp)
        
        # 2. 速度奖励 - 鼓励保持合适速度
        speed_diff = torch.abs(current_speed - target_speed)
        speed_reward = torch.exp(-speed_diff * speed_temp)
        
        # 3. 前进奖励 - 鼓励沿车道前进
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            lane_length = torch.tensor(max(lane.length, 1.0), dtype=torch.float32, device=device)
            progress_ratio = longitudinal / lane_length
            progress_reward = torch.exp(progress_ratio * progress_temp) * 0.1
        else:
            progress_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 4. 动作平滑性奖励 - 惩罚过激动作
        action_magnitude = torch.norm(action)
        action_reward = -torch.exp(action_magnitude * action_temp) + 1.0
        
        # 5. 方向一致性奖励 - 鼓励与车道方向一致
        if hasattr(self.vehicle, 'lane') and self.vehicle.lane:
            lane_heading = torch.tensor(
                self.vehicle.lane.heading_at(longitudinal.item()), 
                dtype=torch.float32, 
                device=device
            )
            heading_diff = torch.abs(vehicle_heading - lane_heading)
            # 归一化角度差到 [0, π]
            pi = torch.tensor(np.pi, dtype=torch.float32, device=device)
            two_pi = torch.tensor(2 * np.pi, dtype=torch.float32, device=device)
            heading_diff = torch.min(heading_diff, two_pi - heading_diff)
            heading_reward = torch.exp(-heading_diff * heading_temp)
        else:
            heading_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 6. 存活时间奖励 - 基础正向奖励
        survival_reward = torch.tensor(0.05, dtype=torch.float32, device=device)
        
        # 7. 碰撞和路面状态（保持原有逻辑）
        collision_reward = torch.tensor(float(self.vehicle.crashed), dtype=torch.float32, device=device)
        on_road_reward = torch.tensor(float(self.vehicle.on_road), dtype=torch.float32, device=device)
        
        # 转换所有结果为float返回
        return {
            "lane_centering_reward": float(lane_centering_reward.item()),
            "speed_reward": float(speed_reward.item()), 
            "progress_reward": float(progress_reward.item()),
            "action_reward": float(action_reward.item()),
            "heading_reward": float(heading_reward.item()),
            "survival_reward": float(survival_reward.item()),
            "collision_reward": float(collision_reward.item()),
            "on_road_reward": float(on_road_reward.item()),
        }

    def _is_terminated(self) -> bool:
        if self.config["terminate_off_road"]:
            return self.vehicle.crashed or not self.vehicle.on_road
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight,
        # Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [100, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, 5],
                [100, 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + 5,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # 3 - Vertical Straight
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [120, -20],
                [120, -30],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [125, -20],
                [125, -30],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + 5,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[5],
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center3,
                radii3,
                np.deg2rad(0),
                np.deg2rad(137),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[5],
            ),
        )

        # 6 - Slant
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [55.7, -15.7],
                [35.7, -35.7],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                [59.3934, -19.2],
                [39.3934, -39.2],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[6],
            ),
        )

        # 7 - Circular Arc #4 - Bugs out when arc is too large, thus 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(315),
                np.deg2rad(170),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(315),
                np.deg2rad(165),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(170),
                np.deg2rad(56),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[7],
            ),
        )
        net.add_lane(
            "h",
            "i",
            CircularLane(
                center4,
                radii4 + 5,
                np.deg2rad(170),
                np.deg2rad(58),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[7],
            ),
        )

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(240),
                np.deg2rad(270),
                width=5,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                speed_limit=speedlimits[8],
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(238),
                np.deg2rad(268),
                width=5,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and
        on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                rand_lane_index = self.road.network.random_lane_index(rng)

                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    rand_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(rand_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)


class RacetrackEnvLarge(RacetrackEnv):
    """
    继承基础环境：扩展了 RacetrackEnv
    更大规模：轨道更长更复杂
    三车道设计：提供更多超车和策略选择机会
    更长直道：从基础版的 58m 扩展到 158m
    复杂转弯：包含更多弯道组合

    credit: @m-walters
    """

    def _make_road(self) -> None:
        net = RoadNetwork()
        w = 5
        w2 = 2 * w
        default_speedlimit = self.config["speed_limit"]

        # Initialise First Lane
        lane = StraightLane(
            [42, 0],
            [200, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=w,
            speed_limit=default_speedlimit,
        )
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w],
                [200, w],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [42, w2],
                [200, w2],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 2 - Circular Arc #1
        center1 = [200, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + w2,
                np.deg2rad(90),
                np.deg2rad(-1),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 3 - Vertical Straight
        delta_extension = -1.0  # Better join
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220, -20],
                [220, -60 + delta_extension],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w, -20],
                [220 + w, -60 + delta_extension],
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [220 + w2, -20],
                [220 + w2, -60 + delta_extension],
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 4 - Circular Arc #2
        center4 = [205, -60]
        radii4 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center4,
                radii4 + w2,
                np.deg2rad(0),
                np.deg2rad(-181),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 5 - Circular Arc #3
        center5 = [170, -60]
        radii5 = 15
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 + 5,
                np.deg2rad(0),
                np.deg2rad(136),
                width=w,
                clockwise=True,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "e",
            "f",
            CircularLane(
                center5,
                radii5 - w,
                np.deg2rad(0),
                np.deg2rad(137),
                width=w,
                clockwise=True,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 6 - Slant
        # Extending [-30,-30]
        extend = np.array([-30, -30])
        start6 = np.array([155.7, -45.7])
        end6 = np.array([135.7, -65.7]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6,
                end6,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_2 = np.array([159.3934, -49.2])
        end6_2 = np.array([139.3934, -69.2]) + extend
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_2,
                end6_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        start6_3 = np.array(
            [
                start6[0] + 2 * (start6_2[0] - start6[0]),
                start6[1] + 2 * (start6_2[1] - start6[1]),
            ]
        )
        end6_3 = np.array(
            [
                end6[0] + 2 * (end6_2[0] - end6[0]),
                end6[1] + 2 * (end6_2[1] - end6[1]),
            ]
        )
        net.add_lane(
            "f",
            "g",
            StraightLane(
                start6_3,
                end6_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 7 - Circular Arc #4
        # Reflect it with the slant
        center7 = np.array([118.1, -48.1]) + extend
        radii7 = 25
        theta7 = 317
        # theta7_end = 270 - (theta7 - 270) - 10
        theta7_end = 205
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end - 3),  # nicer
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + 5,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "g",
            "h",
            CircularLane(
                center7,
                radii7 + w2,
                np.deg2rad(theta7),
                np.deg2rad(theta7_end),
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        # 8 - Next slant
        # Reflected from the last arc's center
        start8 = np.array(
            [
                center7[0] + radii7 * np.cos(np.deg2rad(theta7_end)),
                center7[1] + radii7 * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_2 = np.array(
            [
                center7[0] + (radii7 + w) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w) * np.sin(np.deg2rad(theta7_end)),
            ]
        )
        start8_3 = np.array(
            [
                center7[0] + (radii7 + w2) * np.cos(np.deg2rad(theta7_end)),
                center7[1] + (radii7 + w2) * np.sin(np.deg2rad(theta7_end)),
            ]
        )

        # We preemptively take section 9's radius to make a nice join.
        radii9 = 15
        rad = np.deg2rad(30)
        end8 = np.array([42 - radii9 * np.cos(rad), -radii9 - radii9 * np.sin(rad)])
        end8_2 = np.array(
            [42 - (radii9 + w) * np.cos(rad), -radii9 - (radii9 + w) * np.sin(rad)]
        )
        end8_3 = np.array(
            [42 - (radii9 + w2) * np.cos(rad), -radii9 - (radii9 + w2) * np.sin(rad)]
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8,
                end8,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_2,
                end8_2,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "h",
            "i",
            StraightLane(
                start8_3,
                end8_3,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                width=w,
                speed_limit=default_speedlimit,
            ),
        )

        # 9 - Circular arc 7, end
        # Since y2 = 0...
        center9 = np.array([42, -radii9])
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9,
                np.deg2rad(210),
                np.deg2rad(88),
                width=w,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w,
                np.deg2rad(210),
                np.deg2rad(90),
                width=w,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.STRIPED),
                speed_limit=default_speedlimit,
            ),
        )
        net.add_lane(
            "i",
            "a",
            CircularLane(
                center9,
                radii9 + w2,
                np.deg2rad(212),
                np.deg2rad(88),  # nicer join
                width=w,
                clockwise=False,
                line_types=(LineType.NONE, LineType.CONTINUOUS),
                speed_limit=default_speedlimit,
            ),
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _reward(self, action: torch.Tensor) -> float:
        """
        优化的 RacetrackEnvLarge 奖励函数
        """
        rewards = self._rewards(action)
        
        # 基于配置权重的加权总和
        reward = sum(
            self.config.get(name, 0) * reward_value 
            for name, reward_value in rewards.items()
        )
        
        # 线性映射到 [0, 1] 范围
        reward = utils.lmap(reward, 
                           [self.config["collision_reward"], 2.0], 
                           [0, 1])
        
        # 应用 on_road 乘数
        reward *= rewards["on_road_reward"]
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return float(reward)
    
    def _rewards(self, action: torch.Tensor) -> dict[str, float]:
        """
        多目标奖励组件分解 - 针对 RacetrackEnvLarge 优化
        """
        # 确保 action 是 torch.Tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
    
        # 获取车辆在车道中的位置信息
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
    
        # 转换为 tensor 并确保在同一设备上
        device = action.device if isinstance(action, torch.Tensor) else "cpu"
        longitudinal = torch.tensor(longitudinal, dtype=torch.float32, device=device)
        lateral = torch.tensor(lateral, dtype=torch.float32, device=device)
        current_speed = torch.tensor(self.vehicle.speed, dtype=torch.float32, device=device)
        target_speed = torch.tensor(self.config.get("speed_limit", 10.0), dtype=torch.float32, device=device)
        vehicle_heading = torch.tensor(self.vehicle.heading, dtype=torch.float32, device=device)
        
        # 温度参数 - 针对大型赛道调整
        lateral_temp: float = 2.5  # 更严格的车道居中要求
        speed_temp: float = 1.2    # 速度控制更宽松
        progress_temp: float = 1.5  # 增强前进奖励
        action_temp: float = 1.0   # 动作平滑性
        heading_temp: float = 2.0  # 方向一致性
        efficiency_temp: float = 1.8  # 路径效率
        lane_change_temp: float = 3.0  # 车道变换奖励
        
        # 1. 车道居中奖励 - 在三车道系统中更重要
        lateral_distance = torch.abs(lateral)
        lane_centering_reward = torch.exp(-lateral_distance * lateral_temp)
        
        # 2. 速度奖励 - 鼓励保持合适速度
        speed_diff = torch.abs(current_speed - target_speed)
        speed_reward = torch.exp(-speed_diff * speed_temp)
        
        # 3. 前进奖励 - 鼓励沿车道前进（大型赛道更重要）
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            lane_length = torch.tensor(max(lane.length, 1.0), dtype=torch.float32, device=device)
            progress_ratio = longitudinal / lane_length
            progress_reward = torch.exp(progress_ratio * progress_temp) * 0.15
        else:
            progress_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 4. 动作平滑性奖励 - 惩罚过激动作
        action_magnitude = torch.norm(action)
        action_reward = torch.exp(-action_magnitude * action_temp)
        
        # 5. 方向一致性奖励 - 鼓励与车道方向一致
        if hasattr(self.vehicle, 'lane') and self.vehicle.lane:
            lane_heading = torch.tensor(
                self.vehicle.lane.heading_at(longitudinal.item()), 
                dtype=torch.float32, 
                device=device
            )
            heading_diff = torch.abs(vehicle_heading - lane_heading)
            # 归一化角度差到 [0, π]
            pi = torch.tensor(3.14159, dtype=torch.float32, device=device)
            two_pi = torch.tensor(6.28318, dtype=torch.float32, device=device)
            heading_diff = torch.min(heading_diff, two_pi - heading_diff)
            heading_reward = torch.exp(-heading_diff * heading_temp)
        else:
            heading_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 6. 路径效率奖励 - 奖励高效的路径选择
        if current_speed > 0:
            # 计算前进方向与目标方向的一致性
            forward_direction = torch.tensor([
                torch.cos(vehicle_heading), 
                torch.sin(vehicle_heading)
            ], dtype=torch.float32, device=device)
            
            # 假设目标是沿车道前进
            lane_direction = torch.tensor([
                torch.cos(lane_heading if 'lane_heading' in locals() else vehicle_heading),
                torch.sin(lane_heading if 'lane_heading' in locals() else vehicle_heading)
            ], dtype=torch.float32, device=device)
            
            efficiency_score = torch.dot(forward_direction, lane_direction)
            efficiency_reward = torch.exp(efficiency_score * efficiency_temp) * 0.1
        else:
            efficiency_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 7. 智能车道使用奖励 - 在三车道环境中的策略选择
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            lane_id = self.vehicle.lane_index[2]
            # 中间车道（车道1）用于超车，外侧车道用于正常行驶
            if lane_id == 1:  # 中间车道
                if current_speed > target_speed * 0.8:  # 如果速度较快，奖励使用中间车道
                    lane_strategy_reward = torch.exp(current_speed / target_speed * lane_change_temp) * 0.08
                else:
                    lane_strategy_reward = torch.tensor(-0.02, dtype=torch.float32, device=device)  # 慢速占用超车道的轻微惩罚
            else:  # 外侧车道
                lane_strategy_reward = torch.tensor(0.05, dtype=torch.float32, device=device)  # 基础奖励
        else:
            lane_strategy_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        # 8. 存活时间奖励 - 基础正向奖励
        survival_reward = torch.tensor(0.06, dtype=torch.float32, device=device)
        
        # 9. 碰撞和路面状态
        collision_reward = torch.tensor(-1.0 if self.vehicle.crashed else 0.0, dtype=torch.float32, device=device)
        on_road_reward = torch.tensor(1.0 if self.vehicle.on_road else 0.0, dtype=torch.float32, device=device)
        
        # 10. 完成圈数奖励 - 大型赛道的额外奖励
        completion_bonus = torch.tensor(0.0, dtype=torch.float32, device=device)
        if longitudinal > lane_length * 0.95:  # 接近完成一段轨道
            completion_bonus = torch.tensor(0.2, dtype=torch.float32, device=device)
        
        # 转换所有结果为float返回
        return {
            "lane_centering_reward": float(lane_centering_reward.item()),
            "speed_reward": float(speed_reward.item()), 
            "progress_reward": float(progress_reward.item()),
            "action_reward": float(action_reward.item()),
            "heading_reward": float(heading_reward.item()),
            "efficiency_reward": float(efficiency_reward.item()),
            "lane_strategy_reward": float(lane_strategy_reward.item()),
            "survival_reward": float(survival_reward.item()),
            "collision_reward": float(collision_reward.item()),
            "on_road_reward": float(on_road_reward.item()),
            "completion_bonus": float(completion_bonus.item()),
        }

class RacetrackEnvOval(RacetrackEnv):
    """
    椭圆形状：经典的椭圆赛道设计
    高度可定制：
        可调车道数量（2-7条）
        可调直道长度（100-200m）
        可选路障设置
    战略元素：
        block_lane: 阻挡中间车道
        force_decision: 阻挡第1和第3车道，强制决策
    随机生成：支持随机参数生成更多样的训练场景

    credit: @christophluther
    """

    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["presence", "on_road"],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": False,
                    "lateral": True,
                    "target_speeds": [0, 5, 10],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 300,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1,
                "action_reward": -0.3,
                "controlled_vehicles": 1,
                "other_vehicles": 1,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 10.0,
                "terminate_off_road": True,
                "length": 100,  # 0: random number from [100,200]
                "no_lanes": 3,  # 0: random number from [2,7]
                "block_lane": False,  # block middle lane
                "force_decision": False,  # block 1st and 3rd lane
            }
        )
        return config
    
    def _reward(self, action: torch.Tensor) -> float:
        """
        RacetrackEnvOval 的优化奖励函数
        """
        rewards = self._rewards(action)
        
        # 基于配置权重的加权总和
        reward = sum(
            self.config.get(name, 0) * reward_value 
            for name, reward_value in rewards.items()
        )
        
        # 线性映射到 [0, 1] 范围
        reward = utils.lmap(reward, 
                           [self.config["collision_reward"], 1.5], 
                           [0, 1])
        
        # 应用 on_road 乘数
        reward *= rewards["on_road_reward"]
        reward = np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return float(reward)
    
    def _rewards(self, action: torch.Tensor) -> dict[str, float]:
        """
        多目标奖励组件分解 - 针对 RacetrackEnvOval 优化
        """
        # 确保 action 是 torch.Tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
    
        # 获取车辆在车道中的位置信息
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
    
        # 转换为 tensor 并确保在同一设备上
        device = action.device if isinstance(action, torch.Tensor) else "cpu"
        longitudinal = torch.tensor(longitudinal, dtype=torch.float32, device=device)
        lateral = torch.tensor(lateral, dtype=torch.float32, device=device)
        current_speed = torch.tensor(self.vehicle.speed, dtype=torch.float32, device=device)
        target_speed = torch.tensor(self.config.get("speed_limit", 10.0), dtype=torch.float32, device=device)
        vehicle_heading = torch.tensor(self.vehicle.heading, dtype=torch.float32, device=device)
    
        # 温度参数 - 针对椭圆赛道优化
        lateral_temp: float = 2.2  # 车道居中要求
        speed_temp: float = 1.3    # 速度控制
        progress_temp: float = 1.2  # 前进奖励
        action_temp: float = 0.9   # 动作平滑性
        heading_temp: float = 2.3  # 方向一致性
        efficiency_temp: float = 1.6  # 路径效率
        lap_completion_temp: float = 2.5  # 完成圈数奖励
    
        # 1. 车道居中奖励 - 椭圆赛道中的关键指标
        lateral_distance = torch.abs(lateral)
        lane_centering_reward = torch.exp(-lateral_distance * lateral_temp)
    
        # 2. 速度奖励 - 鼓励保持合适速度
        speed_diff = torch.abs(current_speed - target_speed)
        speed_reward = torch.exp(-speed_diff * speed_temp)
    
        # 3. 前进奖励 - 鼓励沿车道前进
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            lane = self.road.network.get_lane(self.vehicle.lane_index)
            lane_length = torch.tensor(max(lane.length, 1.0), dtype=torch.float32, device=device)
            progress_ratio = longitudinal / lane_length
            progress_reward = torch.exp(progress_ratio * progress_temp) * 0.12
        else:
            progress_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
    
        # 4. 动作平滑性奖励 - 惩罚过激动作，椭圆赛道需要平滑转向
        action_magnitude = torch.norm(action)
        action_reward = torch.exp(-action_magnitude * action_temp)
    
        # 5. 方向一致性奖励 - 鼓励与车道方向一致
        if hasattr(self.vehicle, 'lane') and self.vehicle.lane:
            lane_heading = torch.tensor(
                self.vehicle.lane.heading_at(longitudinal.item()), 
                dtype=torch.float32, 
                device=device
            )
            heading_diff = torch.abs(vehicle_heading - lane_heading)
            # 归一化角度差到 [0, π]
            pi = torch.tensor(3.14159, dtype=torch.float32, device=device)
            two_pi = torch.tensor(6.28318, dtype=torch.float32, device=device)
            heading_diff = torch.min(heading_diff, two_pi - heading_diff)
            heading_reward = torch.exp(-heading_diff * heading_temp)
        else:
            heading_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
    
        # 6. 路径效率奖励 - 奖励高效的路径选择
        if current_speed > 0:
            # 计算速度方向与车道方向的一致性
            speed_direction_consistency = torch.cos(vehicle_heading - lane_heading if 'lane_heading' in locals() else torch.tensor(0.0, device=device))
            efficiency_reward = torch.exp(speed_direction_consistency * efficiency_temp) * 0.08
        else:
            efficiency_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
    
        # 7. 车道策略奖励 - 针对多车道椭圆赛道的智能车道选择
        lane_strategy_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            lane_id = self.vehicle.lane_index[2] if len(self.vehicle.lane_index) > 2 else 0
            no_lanes = self.config.get("no_lanes", 3)
            
            # 鼓励使用内侧车道（通常更高效）
            if no_lanes > 2:
                inner_lane_preference = torch.exp(-torch.tensor(float(lane_id), device=device) * 0.5) * 0.05
                lane_strategy_reward += inner_lane_preference
    
        # 8. 完成圈数/路段奖励 - 椭圆赛道的关键指标
        completion_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index:
            # 检查是否接近完成当前路段
            if longitudinal > lane_length * 0.9:
                completion_reward = torch.exp((longitudinal - lane_length * 0.9) / (lane_length * 0.1) * lap_completion_temp) * 0.15
    
        # 9. 避障奖励 - 针对可能的路障设置
        obstacle_avoidance_reward = torch.tensor(0.05, dtype=torch.float32, device=device)
        if self.config.get("block_lane", False) or self.config.get("force_decision", False):
            # 如果成功避开路障，给予额外奖励
            if self.vehicle.on_road and not self.vehicle.crashed:
                obstacle_avoidance_reward += torch.tensor(0.1, dtype=torch.float32, device=device)
    
        # 10. 存活时间奖励 - 基础正向奖励
        survival_reward = torch.tensor(0.06, dtype=torch.float32, device=device)
    
        # 11. 碰撞和路面状态
        collision_reward = torch.tensor(-1.0 if self.vehicle.crashed else 0.0, dtype=torch.float32, device=device)
        on_road_reward = torch.tensor(1.0 if self.vehicle.on_road else 0.0, dtype=torch.float32, device=device)
    
        # 12. 时间效率奖励 - 鼓励快速完成
        time_efficiency_reward = torch.tensor(0.0, dtype=torch.float32, device=device)
        if hasattr(self, 'time') and hasattr(self, 'config'):
            time_progress = self.time / self.config.get("duration", 300)
            # 奖励在更短时间内的良好表现
            time_progress = torch.tensor(time_progress, dtype=torch.float32, device=device)
            if not self.vehicle.crashed and self.vehicle.on_road:
                time_efficiency_reward = torch.exp(-time_progress * 2.0) * 0.05
    
        # 转换所有结果为float返回
        return {
            "lane_centering_reward": float(lane_centering_reward.item()),
            "speed_reward": float(speed_reward.item()),
            "progress_reward": float(progress_reward.item()),
            "action_reward": float(action_reward.item()),
            "heading_reward": float(heading_reward.item()),
            "efficiency_reward": float(efficiency_reward.item()),
            "lane_strategy_reward": float(lane_strategy_reward.item()),
            "completion_reward": float(completion_reward.item()),
            "obstacle_avoidance_reward": float(obstacle_avoidance_reward.item()),
            "survival_reward": float(survival_reward.item()),
            "collision_reward": float(collision_reward.item()),
            "on_road_reward": float(on_road_reward.item()),
            "time_efficiency_reward": float(time_efficiency_reward.item()),
        }

    def _make_road(self) -> None:
        net = RoadNetwork()

        # define rng
        rng = np.random.default_rng()

        # Set Speed Limits for Road Sections
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # define length,
        if self.config["length"] == 0:
            length = rng.integers(100, high=200)
        else:
            length = self.config["length"]

        if self.config["no_lanes"] == 0:
            no_lanes = rng.integers(2, high=7)
        else:
            no_lanes = self.config["no_lanes"]

        # Lane 1: Initialise First Inner Lane
        lane = StraightLane(
            [0, 0],
            [length + 1, 0],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            width=5,
            speed_limit=speedlimits[1],
        )
        self.lane = lane

        # successively add lanes
        net.add_lane("a", "b", lane)

        # Loop must be separate for every segment to introduce segment names
        for i in range(1, no_lanes - 1):
            # add additional lanes between inner and outer lane
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, i * 5],
                    [length + 1, i * 5],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[1],
                ),
            )

        # Lane 1: Outer Lane
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, (no_lanes - 1) * 5],
                [length + 1, (no_lanes - 1) * 5],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[1],
            ),
        )

        # Turn 1: Inner Lane
        center1 = [length, -20]
        radii1 = 20
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[2],
            ),
        )

        for i in range(1, no_lanes - 1):
            # add additional lanes between inner and outer lane
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + i * 5,
                    np.deg2rad(90),
                    np.deg2rad(0),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[2],
                ),
            )

        # Turn 1: Outer Lane
        net.add_lane(
            "b",
            "c",
            CircularLane(
                center1,
                radii1 + (no_lanes - 1) * 5,
                np.deg2rad(90),
                np.deg2rad(0),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[2],
            ),
        )

        # Vertical Straight 1: Inner Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length + 20, -20],
                [length + 20, -50],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        for i in range(1, no_lanes - 1):
            # add additional lanes between inner and outer lane
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [length + 20 + i * 5, -20],
                    [length + 20 + i * 5, -50],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[3],
                ),
            )

        # Vertical Straight 1: Outer Lane
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [length + 20 + (no_lanes - 1) * 5, -20],
                [length + 20 + (no_lanes - 1) * 5, -50],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[3],
            ),
        )

        # Turn 2: Inner Lane
        center2 = [length + 5, -50]
        radii2 = 15
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[4],
            ),
        )

        for i in range(1, no_lanes - 1):
            # add additional lanes between inner and outer lane
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2 + i * 5,
                    np.deg2rad(0),
                    np.deg2rad(-90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[4],
                ),
            )

        # Turn 2: Outer Lane
        net.add_lane(
            "d",
            "e",
            CircularLane(
                center2,
                radii2 + (no_lanes - 1) * 5,
                np.deg2rad(0),
                np.deg2rad(-90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[4],
            ),
        )

        # Horizontal Straight 2: Inner Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length + 5, -65],
                [-5, -65],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[5],
            ),
        )

        for i in range(1, no_lanes - 1):
            # add additional lanes between inner and outer lane
            net.add_lane(
                "e",
                "f",
                StraightLane(
                    [length + 5, -(65 + i * 5)],
                    [-5, -(65 + i * 5)],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[5],
                ),
            )

        # Horizontal Straight 2: Outer Lane
        net.add_lane(
            "e",
            "f",
            StraightLane(
                [length + 5, -(65 + (no_lanes - 1) * 5)],
                [-5, -(65 + (no_lanes - 1) * 5)],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[5],
            ),
        )

        # Turn 3: Inner Lane
        center4 = [-5, -50]
        radii4 = 15
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[6],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "f",
                "g",
                CircularLane(
                    center4,
                    radii4 + i * 5,
                    np.deg2rad(-90),
                    np.deg2rad(-180),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.NONE),
                    speed_limit=speedlimits[6],
                ),
            )

        # Turn 3: Outer Lane
        net.add_lane(
            "f",
            "g",
            CircularLane(
                center4,
                radii4 + (no_lanes - 1) * 5,
                np.deg2rad(-90),
                np.deg2rad(-180),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[6],
            ),
        )

        # Vertical Straight 2: Inner Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20, -50],
                [-20, -20],
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                width=5,
                speed_limit=speedlimits[7],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "g",
                "h",
                StraightLane(
                    [-20 - i * 5, -50],
                    [-20 - i * 5, -20],
                    line_types=(LineType.STRIPED, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[7],
                ),
            )

        # Vertical Straight 2: Outer Lane
        net.add_lane(
            "g",
            "h",
            StraightLane(
                [-20 - (no_lanes - 1) * 5, -50],
                [-20 - (no_lanes - 1) * 5, -20],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                width=5,
                speed_limit=speedlimits[7],
            ),
        )

        # Turn 4: Inner Lane
        center6 = [0, -20]
        radii6 = 20
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.CONTINUOUS, LineType.NONE),
                speed_limit=speedlimits[8],
            ),
        )

        for i in range(1, no_lanes - 1):
            net.add_lane(
                "h",
                "a",
                CircularLane(
                    center6,
                    radii6 + i * 5,
                    np.deg2rad(180),
                    np.deg2rad(90),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimits[8],
                ),
            )

        # Turn 4: Outer Lane
        net.add_lane(
            "h",
            "a",
            CircularLane(
                center6,
                radii6 + (no_lanes - 1) * 5,
                np.deg2rad(180),
                np.deg2rad(90),
                width=5,
                clockwise=False,
                line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                speed_limit=speedlimits[8],
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        # Scenario to force a "binary" decision
        if self.config["block_lane"]:
            for i in [40, 43, 46, 49]:
                road.objects.append(Obstacle(road, [length - i, 3.75]))
                road.objects.append(Obstacle(road, [length - i, 6.25]))

        if self.config["force_decision"]:
            for i in [-1.25, 1.25, 8.85, 11.25]:
                road.objects.append(Obstacle(road, [length - 90, i]))

        self.road = road

    # CL adjusted to number of lanes
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        # Number of lanes
        no_lanes = self.config["no_lanes"]

        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(no_lanes))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=None, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                rand_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    rand_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(rand_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)
