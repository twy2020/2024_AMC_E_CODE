import pygame
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class Sandbox:
    def __init__(self, config_file="config.json"):
        """
        Initialize the sandbox with layers, configurations, and properties.

        :param config_file: Path to the configuration file.
        """
        # Check if the configuration file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

        # Load configuration
        with open(config_file, "r") as file:
            self.config = json.load(file)

        # Validate essential configuration keys
        required_keys = ["sandbox_width", "sandbox_height", "cell_size", "land_color"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: '{key}'")

        # Initialize sandbox dimensions and cell size
        self.width = self.config["sandbox_width"]
        self.height = self.config["sandbox_height"]
        self.cell_size = self.config["cell_size"]

        # Initialize layers with default values
        self.layers = {
            "Land": np.full((self.height, self.width), self.config["land_default_fertility"], dtype=int),  # Default fertility set to 10000
            "Seedlings": np.zeros((self.height, self.width), dtype=int),  # Seedlings layer
            "MatureTrees": np.zeros((self.height, self.width), dtype=int),  # Mature trees (trunks and leaves)
            "Crops": np.zeros((self.height, self.width), dtype=int),  # Crops layer
            "Insects": np.zeros((self.height, self.width), dtype=int),  # 昆虫图层
            "Birds": np.zeros((self.height, self.width), dtype=int),  # 鸟图层
            "Bats": np.zeros((self.height, self.width), dtype=int),  # 蝙蝠图层
            "Bees": np.zeros((self.height, self.width), dtype=int),  # 蜜蜂图层
            "Worms": np.zeros((self.height, self.width), dtype=int),
            "Herbicide": np.zeros((self.height, self.width), dtype=int),  # 除草剂图层
            "Insecticide": np.zeros((self.height, self.width), dtype=int),  # 除虫剂图层
        }

        # Initialize state tracking for trees and crops
        self.tree_timers = np.zeros((self.height, self.width), dtype=int)  # Seedling growth timers
        self.tree_energies = np.full((self.height, self.width), -1, dtype=int)  # Tree energy levels
        self.crop_timers = np.zeros((self.height, self.width), dtype=int)  # Crop growth/harvest timers
        self.crop_energies = np.full((self.height, self.width), -1, dtype=int)  # Crop energy levels
        self.crop_harvest_energy = 0  # Accumulated energy from harvested crops
        self.insect_energies = np.full((self.height, self.width), -1, dtype=int)  # 昆虫能量
        self.insect_timers = np.zeros((self.height, self.width), dtype=int)  # 昆虫计时器
        self.bird_energies = np.full((self.height, self.width), -1, dtype=int)  # 鸟能量
        self.bird_timers = np.zeros((self.height, self.width), dtype=int)  # 鸟计时器
        self.bird_ages = np.full((self.height, self.width), 0, dtype=int)  # 鸟的年龄
        self.bat_energies = np.full((self.height, self.width), -1, dtype=int)  # 蝙蝠能量
        self.bat_ages = np.zeros((self.height, self.width), dtype=int)  # 蝙蝠年龄
        self.crop_pollinated = np.zeros((self.height, self.width), dtype=bool)  # 标记农作物是否已授粉
        self.season = 0  # 季节编号：0-春季，1-夏季，2-秋季，3-冬季
        self.season_timer = 0  # 记录季节时间
        self.season_length = self.config.get("season_length", 9)  # 每个季节持续时间
        self.herbicide_timers = np.zeros((self.height, self.width), dtype=int)  # 除草剂存在时间计时器
        self.insecticide_timers = np.zeros((self.height, self.width), dtype=int)  # 杀虫剂存在时间计时器
        self.harvested_energy = 0
        self.bee_energies = np.full((self.height, self.width), -1, dtype=int)  # 蜜蜂能量
        self.bee_ages = np.zeros((self.height, self.width), dtype=int)# 蜜蜂寿命
        self.worm_energies = np.full((self.height, self.width), -1, dtype=int)  # 蚯蚓能量

        # Define colors for rendering
        self.colors = {
            "Land": tuple(self.config["land_color"]),  # Convert land color from config
            "Seedling": (0, 255, 0),  # Green for seedlings
            "MatureTree": (0, 100, 0),  # Dark green for mature trees
            "TreeTrunk": (255, 165, 0),  # Orange for tree trunks
            "Crop": (255, 165, 100),  # Orange for crops
            "Insect": (255, 255, 0),  # 昆虫颜色（黄色）
            "Bird": (0, 0, 255),
            "Bat": (255, 0, 255),  # 蝙蝠颜色（紫色）
            "Bee": (255, 192, 203), # 蜜蜂颜色（粉色）
            "Worm": (20, 20, 20) # 蚯蚓颜色（灰色）
        }

        # Layer visibility toggles
        self.layer_visibility = {
            "Land": True,
            "Seedlings": True,
            "MatureTrees": True,
            "Crops": True,
            "Insects": True,  # 昆虫图层可见性
            "Birds": True,
            "Bats": True,
            "Bees": True,
            "Worms": True,
            "Herbicide": True,
            "Insecticide": True
        }

        # Brush for user interactions (default to Seedling placement)
        self.brush = "Seedling"

        # Simulation state tracking
        self.simulating = False  # Simulation starts in a paused state
        self.current_time_step = 0  # Time step counter
        self.stats = []  # List to store simulation statistics per time step

        self.save_folder = "save_history"
        self.history_files = []  # 存储已解析的文件路径
        self.current_file_index = -1  # 当前加载的文件索引

        # 创建保存文件夹
        os.makedirs(self.save_folder, exist_ok=True)
        self.parse_save_files()

        # Logging for initialization success
        print(f"Sandbox initialized with dimensions: {self.width}x{self.height}")
        print(f"Initial fertility set to: {self.layers['Land'][0, 0]}")
        print(f"Layers initialized: {list(self.layers.keys())}")

    def clean_unconnected_leaves(self, start_x, start_y):
        """
        使用递归清理所有未连接到树干的叶子。
        """
        visited = np.zeros_like(self.layers["MatureTrees"], dtype=bool)

        def dfs(x, y):
            if not (0 <= x < self.width and 0 <= y < self.height):
                return
            if visited[y, x] or self.layers["MatureTrees"][y, x] != 2:  # 只处理叶子
                return
            visited[y, x] = True

            # 如果不直接连接到树干，则清除该叶子
            if not any(
                0 <= x + dx < self.width and 0 <= y + dy < self.height and self.layers["MatureTrees"][y + dy, x + dx] == 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ):
                self.layers["MatureTrees"][y, x] = 0

            # 递归检查所有相邻叶子
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(x + dx, y + dy)

        # 清理从起点开始的所有未连接叶子
        dfs(start_x, start_y)

    def simulate_step(self):
        """
        Simulate one time step of the sandbox.
        """
        # Get configuration parameters
        seedling_energy_loss_no_fertility = self.config.get("seedling_energy_loss_no_fertility", 5)
        mature_tree_energy_loss_no_fertility = self.config.get("mature_tree_energy_loss_no_fertility", 15)

        # 季节更替逻辑
        self.season_timer += 1
        if self.season_timer >= self.season_length:
            self.season_timer = 0
            self.season = (self.season + 1) % 4  # 循环更新季节
            print(f"Season changed to: {['Spring', 'Summer', 'Autumn', 'Winter'][self.season]}")
        
        # Seedlings update
        seedling_mask = self.layers["Seedlings"] == 1
        if np.any(seedling_mask):
            # Update timers and fertility
            self.tree_timers[seedling_mask] += 1
            if self.season == 1:  # 夏季
                # 夏季能量增益和土地肥力消耗调整
                summer_energy_gain_multiplier = self.config["summer_energy_gain_multiplier"]
                summer_fertility_cost_multiplier = self.config["summer_fertility_cost_multiplier"]
                
                self.layers["Land"][seedling_mask] = np.maximum(
                    0, 
                    self.layers["Land"][seedling_mask] - self.config["seedling_fertility_cost"] * summer_fertility_cost_multiplier
                )
            else:
                # 非夏季能量调整逻辑
                self.layers["Land"][seedling_mask] = np.maximum(
                    0, 
                    self.layers["Land"][seedling_mask] - self.config["seedling_fertility_cost"]
                )
            
            # 将 self.tree_energies 转换为浮点数类型
            self.tree_energies = self.tree_energies.astype(float)

            # Energy adjustments
            no_fertility_mask = (self.layers["Land"] == 0) & seedling_mask
            with_fertility_mask = (self.layers["Land"] > 0) & seedling_mask

            if self.season == 1:  # 夏季
                # 夏季能量增益和土地肥力消耗调整
                summer_energy_gain_multiplier = self.config["summer_energy_gain_multiplier"]
                summer_fertility_cost_multiplier = self.config["summer_fertility_cost_multiplier"]
                
                # 调整树苗能量增益
                self.tree_energies[no_fertility_mask] -= seedling_energy_loss_no_fertility * summer_energy_gain_multiplier
                self.tree_energies[with_fertility_mask] = np.minimum(
                    self.tree_energies[with_fertility_mask] + self.config["seedling_energy_gain"] * summer_energy_gain_multiplier,
                    self.config["seedling_max_energy"]
                )
                
                # 调整土地肥力消耗
                self.layers["Land"][with_fertility_mask] = np.maximum(
                    0, 
                    self.layers["Land"][with_fertility_mask] - self.config["crop_fertility_cost"] * summer_fertility_cost_multiplier
                )
            else:
                # 非夏季能量调整逻辑
                self.tree_energies[no_fertility_mask] -= seedling_energy_loss_no_fertility
                self.tree_energies[with_fertility_mask] = np.minimum(
                    self.tree_energies[with_fertility_mask] + self.config["seedling_energy_gain"],
                    self.config["seedling_max_energy"]
                )
            
            # Seedlings dying
            dying_seedling_mask = (self.tree_energies < self.config["seedling_min_energy"]) & seedling_mask
            self.layers["Seedlings"][dying_seedling_mask] = 0
            self.layers["Land"][dying_seedling_mask] = np.minimum(
                self.config["land_default_fertility"],
                self.layers["Land"][dying_seedling_mask] + 
                (self.tree_energies[dying_seedling_mask] * self.config["seedling_fertility_return_ratio"]).astype(int)
            )
            self.tree_energies[dying_seedling_mask] = -1
            
            # Seedlings maturing
            maturing_mask = (self.tree_timers >= self.config["seedling_growth_time"]) & seedling_mask
            self.layers["Seedlings"][maturing_mask] = 0
            self.layers["MatureTrees"][maturing_mask] = 1
            self.tree_energies[maturing_mask] = self.config["mature_tree_initial_energy"]
        
        # Mature trees update
        trunk_mask = self.layers["MatureTrees"] == 1
        if np.any(trunk_mask):
            # Update fertility and energy
            if self.season == 1:  # 夏季
                # 夏季能量增益和土地肥力消耗调整
                summer_energy_gain_multiplier = self.config["summer_energy_gain_multiplier"]
                summer_fertility_cost_multiplier = self.config["summer_fertility_cost_multiplier"]
                
                self.layers["Land"][trunk_mask] = np.maximum(
                    0, 
                    self.layers["Land"][trunk_mask] - self.config["mature_tree_fertility_cost"] * summer_fertility_cost_multiplier
                )

                # Split trunks into two cases: with and without fertility
                no_fertility_trunk = (self.layers["Land"] == 0) & trunk_mask
                with_fertility_trunk = (self.layers["Land"] > 0) & trunk_mask

                # Energy gain or loss based on fertility
                self.tree_energies[no_fertility_trunk] = np.maximum(
                    self.tree_energies[no_fertility_trunk] - self.config["mature_tree_energy_loss_no_fertility"] * summer_energy_gain_multiplier, 
                    0
                )
                self.tree_energies[with_fertility_trunk] = np.minimum(
                    self.tree_energies[with_fertility_trunk] + self.config["mature_tree_energy_gain"] * summer_energy_gain_multiplier,
                    self.config["mature_tree_max_energy"]
                )
            else:
                # 非夏季能量调整逻辑
                self.layers["Land"][trunk_mask] = np.maximum(
                    0, 
                    self.layers["Land"][trunk_mask] - self.config["mature_tree_fertility_cost"]
                )

                # Split trunks into two cases: with and without fertility
                no_fertility_trunk = (self.layers["Land"] == 0) & trunk_mask
                with_fertility_trunk = (self.layers["Land"] > 0) & trunk_mask

                # Energy gain or loss based on fertility
                self.tree_energies[no_fertility_trunk] = np.maximum(
                    self.tree_energies[no_fertility_trunk] - self.config["mature_tree_energy_loss_no_fertility"], 
                    0
                )
                self.tree_energies[with_fertility_trunk] = np.minimum(
                    self.tree_energies[with_fertility_trunk] + self.config["mature_tree_energy_gain"],
                    self.config["mature_tree_max_energy"]
                )

            # Handle trunk energy loss and death
            dying_trunk_mask = (self.tree_energies < self.config["mature_tree_min_energy"]) & trunk_mask

            # Add lifetime check: trees die after reaching maximum lifetime
            lifetime_exceeded_mask = (self.tree_timers >= self.config["mature_tree_lifetime"]) & trunk_mask
            total_dying_trunk_mask = dying_trunk_mask | lifetime_exceeded_mask

            # Remove dying trunks
            self.layers["MatureTrees"][total_dying_trunk_mask] = 0

            # Return fertility to the land for dying trees
            self.layers["Land"][total_dying_trunk_mask] = np.minimum(
                self.config["land_default_fertility"],
                self.layers["Land"][total_dying_trunk_mask] + 
                (self.tree_energies[total_dying_trunk_mask] * self.config["mature_tree_fertility_return_ratio"]).astype(int)
            )

            # Mark tree energies of the dying trunks as invalid
            self.tree_energies[total_dying_trunk_mask] = -1

            # Add leaves around trunk
            if np.any(trunk_mask):  # Check for the existence of trunks
                trunk_positions = np.argwhere(trunk_mask)  # Get all trunk positions ([y, x])
                for ty, tx in trunk_positions:  # Iterate over each trunk position
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check neighboring cells
                        nx, ny = tx + dx, ty + dy  # Calculate neighboring position
                        if 0 <= nx < self.width and 0 <= ny < self.height:  # Ensure valid position
                            if (
                                self.layers["MatureTrees"][ny, nx] == 0 and  # Not another trunk or leaf
                                self.layers["Seedlings"][ny, nx] == 0 and   # Not a seedling
                                self.layers["Crops"][ny, nx] == 0           # Not a crop
                            ):
                                self.layers["MatureTrees"][ny, nx] = 2  # Mark as leaf

            # 秋季树叶掉落逻辑
            if self.season == 2:  # 检查是否为秋季
                leaf_positions = np.argwhere(self.layers["MatureTrees"] == 2)  # 获取所有树叶位置
                for ly, lx in leaf_positions:
                    if np.random.rand() < self.config["autumn_leaf_drop_chance"]:  # 按概率判断是否掉落
                        self.layers["MatureTrees"][ly, lx] = 0  # 移除树叶
                        self.layers["Land"][ly, lx] = min(
                            self.config["land_default_fertility"],
                            self.layers["Land"][ly, lx] + self.config["autumn_leaf_fertility_return"]
                        )

            # Clear unconnected leaves
            dying_positions = np.argwhere(total_dying_trunk_mask)  # Get the positions of the dying trunks
            for dy, dx in dying_positions:
                for dy_offset, dx_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = dy + dy_offset, dx + dx_offset
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # If the adjacent cell is a leaf, check if it's still connected to any trunk
                        if self.layers["MatureTrees"][ny, nx] == 2:  # Leaf
                            self.clean_unconnected_leaves(nx, ny)

            # Increment timers for all trunks
            self.tree_timers[trunk_mask] += 1
            
        # Reproduction logic
        if self.season == 0:  # 春季
            reproduction_mask = trunk_mask & (np.random.rand(*trunk_mask.shape) < self.config["mature_tree_reproduction_chance"] * self.config["spring_reproduction_multiplier"])
        elif self.season == 2:
            reproduction_mask = trunk_mask & (np.random.rand(*trunk_mask.shape) < self.config["mature_tree_reproduction_chance"] * self.config["autumn_reproduction_penalty"])
        elif self.season == 3:
            reproduction_mask = trunk_mask & (np.random.rand(*trunk_mask.shape) < self.config["mature_tree_reproduction_chance"] * self.config["winter_reproduction_penalty"])
        else: 
            reproduction_mask = trunk_mask & (np.random.rand(*trunk_mask.shape) < self.config["mature_tree_reproduction_chance"])

        if np.any(reproduction_mask):
            # 获取所有符合繁殖条件的树干位置
            trunk_positions = np.argwhere(reproduction_mask)  # 注意返回 [y, x]
            for ty, tx in trunk_positions:  # 遍历每个符合条件的树干位置
                # 在树干周围的 5x5 区域生成候选位置，排除中心位置
                potential_positions = [
                    (tx + dx, ty + dy)
                    for dx in range(-2, 3) for dy in range(-2, 3)
                    if (dx != 0 or dy != 0) and 0 <= tx + dx < self.width and 0 <= ty + dy < self.height
                ]
                
                # 随机选择一个候选位置
                if potential_positions:
                    nx, ny = potential_positions[np.random.randint(len(potential_positions))]
                    # 检查选中的位置是否为空地
                    if self.layers["MatureTrees"][ny, nx] == 0 and self.layers["Crops"][ny, nx] == 0 and self.layers["Seedlings"][ny, nx] == 0:
                        # 如果位置为空地，生成种子
                        self.layers["Seedlings"][ny, nx] = 1
                        self.tree_timers[ny, nx] = 0
                        self.tree_energies[ny, nx] = self.config["seedling_initial_energy"]
                    # 如果不是空地，放弃本次生成机会
        
        # 农作物更新逻辑
        crop_mask = self.layers["Crops"] == 1  # 包括所有农作物单元
        if np.any(crop_mask):
            for y, x in np.argwhere(crop_mask):
                # 增加计时器
                self.crop_timers[y, x] += 1

                # 季节性能量调整和土地肥力消耗
                if self.season == 1:  # 夏季
                    summer_energy_gain_multiplier = self.config["summer_energy_gain_multiplier"]
                    summer_fertility_cost_multiplier = self.config["summer_fertility_cost_multiplier"]

                    # 调整土地肥力消耗
                    self.layers["Land"][y, x] = max(
                        0,
                        self.layers["Land"][y, x] - self.config["crop_fertility_cost"] * summer_fertility_cost_multiplier
                    )
                    # 调整能量获取
                    if self.layers["Land"][y, x] > 0:
                        self.crop_energies[y, x] = min(
                            self.crop_energies[y, x] + self.config["crop_energy_gain_with_fertility"] * summer_energy_gain_multiplier,
                            self.config["crop_max_energy"]
                        )
                    else:
                        self.crop_energies[y, x] += self.config["crop_energy_gain_no_fertility"] * summer_energy_gain_multiplier
                else:  # 非夏季逻辑
                    self.layers["Land"][y, x] = max(
                        0,
                        self.layers["Land"][y, x] - self.config["crop_fertility_cost"]
                    )
                    if self.layers["Land"][y, x] > 0:
                        self.crop_energies[y, x] = min(
                            self.crop_energies[y, x] + self.config["crop_energy_gain_with_fertility"],
                            self.config["crop_max_energy"]
                        )
                    else:
                        self.crop_energies[y, x] += self.config["crop_energy_gain_no_fertility"]

                # 确保作物能量不会低于 1
                self.crop_energies[y, x] = max(self.crop_energies[y, x], 1)

                # 检查是否达到采收周期
                if self.crop_timers[y, x] >= self.config["crop_harvest_time"]:
                    # 计算本周期收获的总能量
                    harvested_energy = int(
                        self.crop_energies[y, x] * self.config["crop_harvest_energy_ratio"]
                    )
                    self.harvested_energy += harvested_energy  # 累积到总收获能量

                    # 更新土地肥力
                    self.layers["Land"][y, x] += int(
                        self.config["crop_fertility_return_ratio_harvested"] * self.crop_energies[y, x]
                    )

                    # 重置作物状态
                    self.crop_energies[y, x] = self.config["crop_initial_energy"]  # 重置能量
                    self.crop_timers[y, x] = 0  # 重置计时器
                    self.crop_pollinated[y, x] = False  # 重置授粉状态

        # 昆虫自然生成逻辑
        if self.season == 1 and np.random.rand() < self.config["summer_insect_spawn_chance"]:
            spawn_positions = np.argwhere((self.layers["MatureTrees"] == 1) | (self.layers["Crops"] == 1))
            if len(spawn_positions) > 0:
                ny, nx = spawn_positions[np.random.randint(len(spawn_positions))]
                if self.layers["Insects"][ny, nx] == 0:  # 确保目标格子没有昆虫
                    self.layers["Insects"][ny, nx] = 1
                    self.insect_energies[ny, nx] = self.config["insect_initial_energy"]

        # 鸟自然生成逻辑
        if self.season == 0 and np.random.rand() < self.config["spring_bird_spawn_chance"]:
            spawn_positions = np.argwhere(self.layers["MatureTrees"] == 2)
            if len(spawn_positions) > 0:
                ny, nx = spawn_positions[np.random.randint(len(spawn_positions))]
                if self.layers["Birds"][ny, nx] == 0 and self.layers["Bats"][ny, nx] == 0 and self.layers["Bees"][ny, nx] == 0:  # 确保目标格子没有鸟
                    self.layers["Birds"][ny, nx] = 1
                    self.bird_energies[ny, nx] = self.config["bird_initial_energy"]
                    self.bird_ages[ny, nx] = 0

        # 蜜蜂春季刷新逻辑
        if self.season == 0:  # 春季
            for ty, tx in np.argwhere(self.layers["MatureTrees"] == 1):  # 仅考虑树干
                if np.random.rand() < self.config["bee_spring_spawn_chance"]:
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = ty + dy, tx + dx
                            if (
                                0 <= nx < self.width and 0 <= ny < self.height and
                                self.layers["Bees"][ny, nx] == 0 and self.layers["Insects"][ny, nx] == 0
                            ):
                                self.layers["Bees"][ny, nx] = 1
                                self.bee_energies[ny, nx] = self.config["bee_initial_energy"]
                                self.bee_ages[ny, nx] = 0
                                break

        # 蚯蚓春季生成逻辑
        if self.season == 0:  # 春季
            for y in range(self.height):
                for x in range(self.width):
                    if (
                        self.layers["Land"][y, x] > 1000 and  # 土地肥力足够
                        self.layers["Worms"][y, x] == 0 and  # 当前格子无蚯蚓
                        np.random.rand() < self.config["worm_spring_spawn_chance"]  # 生成概率
                    ):
                        # 确保附近没有蚯蚓
                        nearby_worms = [
                            (y + dy, x + dx)
                            for dy in range(-self.config["worm_search_radius"], self.config["worm_search_radius"] + 1)
                            for dx in range(-self.config["worm_search_radius"], self.config["worm_search_radius"] + 1)
                            if 0 <= x + dx < self.width and 0 <= y + dy < self.height and self.layers["Worms"][y + dy, x + dx] == 1
                        ]
                        if not nearby_worms:
                            self.layers["Worms"][y, x] = 1
                            self.worm_energies[y, x] = self.config["worm_initial_energy"]

        # 昆虫更新逻辑
        insect_mask = self.layers["Insects"] == 1
        if np.any(insect_mask):
            for y, x in np.argwhere(insect_mask):
                energy = self.insect_energies[y, x]
                
                # 如果是冬天，额外增加死亡几率
                if self.season == 3 and np.random.rand() < self.config["winter_death_chance"] * 3:
                    self.layers["Insects"][y, x] = 0
                    continue  # 直接跳过后续逻辑

                # 随机移动逻辑
                if np.random.rand() < 0.8:
                    crop_positions = np.argwhere(self.layers["Crops"] == 1)
                    if len(crop_positions) > 0:
                        target = crop_positions[np.random.randint(len(crop_positions))]
                        dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                    else:
                        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                else:
                    dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])

                # 移动目标位置
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:  # 确保目标合法
                    energy -= self.config["insect_energy_loss_per_move"]  # 移动能量消耗

                    # 如果新位置有农作物，则根据农作物能量决定捕食概率
                    if self.layers["Crops"][ny, nx] == 1:
                        crop_energy = self.crop_energies[ny, nx]

                        # 示例：捕食概率随农作物能量线性上升，并设定一个上限
                        # 你可根据需求修改基准值(insect_base_feed_probability)和归一化因子(normalize_factor)
                        insect_base_feed_probability = self.config.get("insect_base_feed_probability", 0.3)  # 基准捕食概率
                        normalize_factor = self.config.get("crop_energy_normalize_factor", 200.0)  # 用于归一化crop_energy
                        feed_probability = insect_base_feed_probability * (crop_energy / normalize_factor)

                        # 将捕食概率限定在[0, 1]之间
                        feed_probability = min(max(feed_probability, 0.0), 1.0)

                        # 根据feed_probability决定是否捕食
                        if np.random.rand() < feed_probability:
                            energy_gain = crop_energy * self.config["insect_crop_energy_gain_ratio"]
                            energy += energy_gain
                            # 农作物被部分消耗后下降能量
                            self.crop_energies[ny, nx] *= 1 - self.config["insect_crop_energy_gain_ratio"]

                    # 昆虫繁殖逻辑
                    insect_reproduction_chance = self.config["insect_reproduction_chance"]
                    if self.season == 2:
                        insect_reproduction_chance *= self.config["autumn_reproduction_penalty"]
                    elif self.season == 3:
                        insect_reproduction_chance *= self.config["winter_reproduction_penalty"]

                    if energy > self.config["insect_reproduction_energy"] and np.random.rand() < insect_reproduction_chance:
                        # 寻找周围可放置新昆虫的空位
                        potential_positions = [
                            (nx + dx, ny + dy)
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if 0 <= nx + dx < self.width and 0 <= ny + dy < self.height and
                            self.layers["Insects"][ny + dy, nx + dx] == 0
                        ]
                        if potential_positions:
                            bx, by = potential_positions[np.random.randint(len(potential_positions))]
                            self.layers["Insects"][by, bx] = 1
                            self.insect_energies[by, bx] = self.config["insect_initial_energy"]

                    # 昆虫死亡逻辑
                    if energy < self.config["insect_death_energy_threshold"]:
                        # 昆虫死亡，土地获得一定肥力回馈
                        self.layers["Insects"][y, x] = 0
                        self.layers["Land"][y, x] = min(
                            self.config["land_default_fertility"],
                            self.layers["Land"][y, x] + self.config["insect_fertility_return"]
                        )
                    else:
                        # 更新昆虫位置和能量
                        self.layers["Insects"][ny, nx] = 1
                        self.insect_energies[ny, nx] = energy
                        self.layers["Insects"][y, x] = 0  # 清除原位置昆虫

        # 鸟单位更新逻辑
        bird_mask = self.layers["Birds"] == 1
        if np.any(bird_mask):
            for y, x in np.argwhere(bird_mask):
                energy = self.bird_energies[y, x]
                age = self.bird_ages[y, x]

                # 增加年龄
                age += 1
                self.bird_ages[y, x] = age

                # 检查寿命限制
                if age > self.config["bird_lifetime"]:
                    self.layers["Birds"][y, x] = 0
                    self.layers["Land"][y, x] += int(energy * self.config["bird_fertility_return_ratio"])
                    continue

                if self.season == 3 and np.random.rand() < self.config["winter_death_chance"]:
                    self.layers["Birds"][y, x] = 0
                    continue

                # 随机移动逻辑
                move_type = np.random.rand()
                if move_type < 0.6:  # 80% 捕食昆虫
                    insect_positions = np.argwhere(self.layers["Insects"] == 1)
                    if len(insect_positions) > 0:
                        target = insect_positions[np.random.randint(len(insect_positions))]
                        dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                    else:
                        dx, dy = np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])
                elif move_type < 0.6:  # 10% 捕食农作物
                    crop_positions = np.argwhere(self.layers["Crops"] == 1)
                    if len(crop_positions) > 0:
                        target = crop_positions[np.random.randint(len(crop_positions))]
                        dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                    else:
                        dx, dy = np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])
                else:  # 10% 捕食树叶、树干、树苗
                    tree_positions = np.argwhere((self.layers["MatureTrees"] > 0) | (self.layers["Seedlings"] > 0))
                    if len(tree_positions) > 0:
                        target = tree_positions[np.random.randint(len(tree_positions))]
                        dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                    else:
                        dx, dy = np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:  # 确保目标合法
                    energy -= self.config["bird_energy_loss_per_move"]

                    # 捕食昆虫
                    for dx_offset, dy_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        tx, ty = nx + dx_offset, ny + dy_offset
                        if 0 <= tx < self.width and 0 <= ty < self.height:
                            if self.layers["Insects"][ty, tx] == 1:
                                insect_energy = self.insect_energies[ty, tx]
                                energy += int(insect_energy * self.config["bird_insect_energy_gain_ratio"])
                                self.layers["Insects"][ty, tx] = 0

                    # 捕食农作物
                    if self.layers["Crops"][ny, nx] == 1:
                        self.crop_energies[ny, nx] = max(0, self.crop_energies[ny, nx] - 20)
                        energy += 10

                    # 捕食树干
                    if self.layers["MatureTrees"][ny, nx] > 0:
                        tree_energy = self.tree_energies[ny, nx]
                        energy += int(tree_energy * self.config["bird_tree_energy_gain_ratio"])
                        self.tree_timers[ny, nx] += 2  # 增加树干寿命5点

                    # 鸟单位繁殖逻辑
                    bird_reproduction_chance = self.config["bird_reproduction_chance"]
                    if self.season == 0:
                        bird_reproduction_chance *= self.config["spring_reproduction_multiplier"]
                    elif self.season == 2:
                        bird_reproduction_chance *= self.config["autumn_reproduction_penalty"]
                    elif self.season == 3:
                        bird_reproduction_chance *= self.config["winter_reproduction_penalty"]

                    if energy > 1000 and np.random.rand() < bird_reproduction_chance:
                        tree_positions = np.argwhere(
                            (self.layers["MatureTrees"] == 1) | (self.layers["MatureTrees"] == 2)
                        )  # 树干或树叶
                        potential_positions = [
                            (tx, ty) for ty, tx in tree_positions
                            if abs(tx - x) <= 15 and abs(ty - y) <= 15
                        ]  # 选取9x9区域内的树干或树叶
                        if potential_positions:
                            target = potential_positions[np.random.randint(len(potential_positions))]
                            bx, by = target
                            if self.layers["Birds"][by, bx] == 0:
                                self.layers["Birds"][by, bx] = 1
                                self.bird_energies[by, bx] = self.config["bird_initial_energy"]
                                self.bird_ages[by, bx] = 0

                    # 更新鸟位置和能量
                    if energy >= self.config["bird_death_energy_threshold"]:
                        self.layers["Birds"][ny, nx] = 1
                        self.bird_energies[ny, nx] = energy
                        self.layers["Birds"][y, x] = 0
                    else:  # 能量不足死亡
                        self.layers["Birds"][y, x] = 0
                        self.layers["Land"][y, x] += int(energy * self.config["bird_fertility_return_ratio"])

        # 蝙蝠更新逻辑
        bat_mask = self.layers["Bats"] == 1
        if np.any(bat_mask):
            for y, x in np.argwhere(bat_mask):
                energy = self.bat_energies[y, x]
                age = self.bat_ages[y, x]

                # 增加年龄
                age += 1
                self.bat_ages[y, x] = age

                # 检查寿命限制
                if age > self.config["bat_lifetime"]:
                    self.layers["Bats"][y, x] = 0
                    self.layers["Land"][y, x] += int(energy * self.config["bat_fertility_return_ratio"])
                    continue

                if self.season == 3 and np.random.rand() < self.config["winter_death_chance"]:
                    self.layers["Bats"][y, x] = 0

                # 随机移动逻辑
                insect_positions = np.argwhere(self.layers["Insects"] == 1)
                if len(insect_positions) > 0 and np.random.rand() < 0.9:
                    target = insect_positions[np.random.randint(len(insect_positions))]
                    dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                else:
                    dx, dy = np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])

                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:  # 确保目标合法
                    energy -= self.config["bat_energy_loss_per_move"]

                    # 捕食昆虫
                    for dx_offset, dy_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        tx, ty = nx + dx_offset, ny + dy_offset
                        if 0 <= tx < self.width and 0 <= ty < self.height:
                            if self.layers["Insects"][ty, tx] == 1:
                                insect_energy = self.insect_energies[ty, tx]
                                energy += int(insect_energy * self.config["bat_insect_energy_gain_ratio"])
                                self.layers["Insects"][ty, tx] = 0

                    # 授粉逻辑
                    if self.layers["Crops"][ny, nx] == 1 and not self.crop_pollinated[ny, nx]:
                        self.crop_energies[ny, nx] *= self.config["bat_pollination_boost_multiplier"]
                        self.crop_pollinated[ny, nx] = True  # 标记为已授粉

                    # 蝙蝠繁殖逻辑
                    bat_reproduction_chance = self.config["bat_reproduction_chance"]
                    if self.season == 0:
                        bat_reproduction_chance *= self.config["spring_reproduction_multiplier"]
                    elif self.season == 2:
                        bat_reproduction_chance *= self.config["autumn_reproduction_penalty"]
                    elif self.season == 3:
                        bat_reproduction_chance *= self.config["winter_reproduction_penalty"]

                    if energy > 1000 and np.random.rand() < bat_reproduction_chance:
                        tree_positions = np.argwhere(
                            (self.layers["MatureTrees"] == 1) | (self.layers["MatureTrees"] == 2)
                        )  # 树干或树叶
                        potential_positions = [
                            (tx, ty) for ty, tx in tree_positions
                            if abs(tx - x) <= 20 and abs(ty - y) <= 20
                        ]  # 选取9x9区域内的树干或树叶
                        if potential_positions:
                            target = potential_positions[np.random.randint(len(potential_positions))]
                            bx, by = target
                            if self.layers["Bats"][by, bx] == 0:
                                self.layers["Bats"][by, bx] = 1
                                self.bat_energies[by, bx] = self.config["bat_initial_energy"]
                                self.bat_ages[by, bx] = 0

                    # 更新蝙蝠位置和能量
                    if energy >= self.config["bat_death_energy_threshold"]:
                        self.layers["Bats"][ny, nx] = 1
                        self.bat_energies[ny, nx] = energy
                        self.layers["Bats"][y, x] = 0
                    else:  # 能量不足死亡
                        self.layers["Bats"][y, x] = 0
                        self.layers["Land"][y, x] += int(energy * self.config["bat_fertility_return_ratio"])

        # 蜜蜂更新逻辑
        bee_mask = self.layers["Bees"] == 1
        if np.any(bee_mask):
            for y, x in np.argwhere(bee_mask):
                energy = self.bee_energies[y, x]
                age = self.bee_ages[y, x]

                # 增加年龄
                age += 1
                self.bee_ages[y, x] = age

                # 检查寿命限制
                if age > self.config["bee_lifetime"]:
                    self.layers["Bees"][y, x] = 0
                    self.layers["Land"][y, x] += int(energy * self.config["bee_fertility_return_ratio"])
                    continue

                # 冬季高概率死亡
                if self.season == 3 and np.random.rand() < self.config["bee_winter_death_chance"]:
                    self.layers["Bees"][y, x] = 0
                    self.layers["Land"][y, x] += int(energy * self.config["bee_fertility_return_ratio"])
                    continue

                # 搜索农作物范围
                target_found = False
                crop_positions = []
                for dy in range(-self.config["bee_search_range"], self.config["bee_search_range"] + 1):
                    for dx in range(-self.config["bee_search_range"], self.config["bee_search_range"] + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= nx < self.width and 0 <= ny < self.height and self.layers["Crops"][ny, nx] == 1:
                            crop_positions.append((ny, nx))

                # 如果找到目标，向目标移动
                if crop_positions:
                    target = crop_positions[np.random.randint(len(crop_positions))]
                    dx, dy = np.sign(target[1] - x), np.sign(target[0] - y)
                    nx, ny = x + dx, y + dy
                    target_found = True
                else:
                    # 随机移动逻辑
                    dx, dy = np.random.choice([-2, -1, 0, 1, 2]), np.random.choice([-2, -1, 0, 1, 2])
                    nx, ny = x + dx, y + dy

                # 确保目标合法，并避免与昆虫冲突
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.layers["Insects"][ny, nx] == 0:  # 避免昆虫冲突
                        energy -= self.config["bee_energy_loss_per_move"]

                        # 采蜜机制
                        if self.layers["Crops"][ny, nx] == 1:
                            energy += self.config["bee_energy_gain_from_crops"]
                            self.crop_energies[ny, nx] *= self.config["bee_pollination_boost_multiplier"]

                        # 更新蜜蜂位置和能量
                        if energy > 0:
                            self.layers["Bees"][ny, nx] = 1
                            self.bee_energies[ny, nx] = energy
                            self.layers["Bees"][y, x] = 0
                        else:  # 能量不足死亡
                            self.layers["Bees"][y, x] = 0

        # 蚯蚓更新逻辑
        worm_mask = self.layers["Worms"] == 1
        if np.any(worm_mask):
            for y, x in np.argwhere(worm_mask):
                energy = self.worm_energies[y, x]

                # 检查寿命限制或能量耗尽
                if energy <= 0:
                    self.layers["Worms"][y, x] = 0
                    continue

                # 土壤肥力提升
                self.layers["Land"][y, x] = min(
                    self.config["land_default_fertility"],
                    self.layers["Land"][y, x] + self.config["worm_soil_fertility_boost"]
                )

                # 随机选择一个方向
                dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                nx, ny = x + dx, y + dy

                # 确保目标位置合法
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # 如果目标位置有鸟，可能被捕食
                    if self.layers["Birds"][ny, nx] == 1 and np.random.rand() < self.config["worm_energy_gain_by_bird"]:
                        bird_energy_gain = int(energy * self.config["worm_energy_gain_by_bird"])
                        self.bird_energies[ny, nx] += bird_energy_gain
                        self.layers["Worms"][y, x] = 0  # 蚯蚓死亡
                        continue

                    # 消耗能量并移动
                    energy -= self.config["worm_energy_loss_per_move"]
                    if energy > 0:
                        self.layers["Worms"][ny, nx] = 1
                        self.worm_energies[ny, nx] = energy
                        self.layers["Worms"][y, x] = 0
                    else:  # 能量不足死亡
                        self.layers["Worms"][y, x] = 0

        # 除草剂更新逻辑
        herbicide_mask = self.layers["Herbicide"] == 1
        if np.any(herbicide_mask):
            for y, x in np.argwhere(herbicide_mask):
                # 增加计时器
                self.herbicide_timers[y, x] += 1

                # 清除树木和树苗，不返还肥力
                if self.layers["MatureTrees"][y, x] > 0 or self.layers["Seedlings"][y, x] > 0:
                    self.layers["MatureTrees"][y, x] = 0
                    self.layers["Seedlings"][y, x] = 0

                # 禁止肥力增加，同时每时间步减少肥力
                self.layers["Land"][y, x] = max(0, self.layers["Land"][y, x] - self.config["herbicide_fertility_penalty"])

                # 增加农作物能量加成
                if self.layers["Crops"][y, x] == 1:
                    self.crop_energies[y, x] += self.config["herbicide_crop_energy_boost"]

                # 判断是否超过存在周期
                if self.herbicide_timers[y, x] >= self.config["herbicide_lifetime"]:
                    self.layers["Herbicide"][y, x] = 0  # 失效
                    self.herbicide_timers[y, x] = -self.config["herbicide_generate_period"]  # 开始计时以重新生成

        # 重新生成除草剂
        herbicide_generate_mask = (self.herbicide_timers < 0)
        if np.any(herbicide_generate_mask):
            self.herbicide_timers[herbicide_generate_mask] += 1  # 计时增加
            for y, x in np.argwhere(herbicide_generate_mask):
                if self.herbicide_timers[y, x] == 0:  # 计时完成后重新激活
                    self.layers["Herbicide"][y, x] = 1

        # 杀虫剂更新逻辑
        insecticide_mask = self.layers["Insecticide"] == 1
        if np.any(insecticide_mask):
            for y, x in np.argwhere(insecticide_mask):
                # 增加计时器
                self.insecticide_timers[y, x] += 1

                # 杀死昆虫
                if self.layers["Insects"][y, x] == 1:
                    self.layers["Insects"][y, x] = 0
                if self.layers["Bees"][y, x] == 1:
                    self.layers["Bees"][y, x] = 0

                # 判断是否超过存在周期
                if self.insecticide_timers[y, x] >= self.config["insecticide_lifetime"]:
                    self.layers["Insecticide"][y, x] = 0  # 失效
                    self.insecticide_timers[y, x] = -self.config["insecticide_generate_period"]  # 开始计时以重新生成

        # 重新生成杀虫剂
        insecticide_generate_mask = (self.insecticide_timers < 0)
        if np.any(insecticide_generate_mask):
            self.insecticide_timers[insecticide_generate_mask] += 1  # 计时增加
            for y, x in np.argwhere(insecticide_generate_mask):
                if self.insecticide_timers[y, x] == 0:  # 计时完成后重新激活
                    self.layers["Insecticide"][y, x] = 1

        # 统计各单位数量
        crop_count = np.count_nonzero(self.layers["Crops"])
        insect_count = np.count_nonzero(self.layers["Insects"])
        bird_count = np.count_nonzero(self.layers["Birds"])
        bat_count = np.count_nonzero(self.layers["Bats"])
        bee_count = np.count_nonzero(self.layers["Bees"])
        worm_count = np.count_nonzero(self.layers["Worms"])

        # 计算能量收获比
        total_crop_energy = self.config["crop_max_energy"] * crop_count
        harvest_energy_ratio = (
            self.harvested_energy / total_crop_energy if total_crop_energy > 0 else 0
        )

        # 人类活动区土地平均肥力（农作物 + 农药区域）
        human_activity_mask = (
            (self.layers["Crops"] == 1) | (self.layers["Herbicide"] == 1) | (self.layers["Insecticide"] == 1)
        )
        avg_human_activity_fertility = (
            np.sum(self.layers["Land"][human_activity_mask]) / np.count_nonzero(human_activity_mask)
            if np.count_nonzero(human_activity_mask) > 0
            else 0
        )

        # 记录统计数据
        self.stats.append(
            (self.current_time_step, crop_count, insect_count, bird_count, bat_count, bee_count, worm_count, harvest_energy_ratio, avg_human_activity_fertility)
        )

        # 重置收获能量
        self.harvested_energy = 0

        # 更新时间步
        self.current_time_step += 1

    # 更新保存功能
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = f"./result/simulation_{timestamp}.xlsx"
        config_file = f"./result/simulation_{timestamp}.txt"

        # 保存Excel文件
        df = pd.DataFrame(
            self.stats,
            columns=["TimeStep", "Crops", "Insects", "Birds", "Bats", "Bees", "Worms", "HarvestEnergyRatio", "AvgHumanActivityFertility"]
        )
        df.to_excel(excel_file, index=False)

        # 保存配置到文本文件
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"Results saved to {excel_file} and {config_file}.")


    def plot_stats(self):
        # 提取统计数据
        timesteps, crops, insects, birds, bats, bees, worms, harvest_energy_ratio, avg_human_activity_fertility = zip(*self.stats)

        # 创建第一个窗口：显示动物单位数量
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, insects, label="Insects", linestyle='-', marker='o', color='yellow')
        plt.plot(timesteps, birds, label="Birds", linestyle='-', marker='s', color='blue')
        plt.plot(timesteps, bats, label="Bats", linestyle='-', marker='^', color='purple')
        plt.plot(timesteps, bees, label="Bees", linestyle='-', marker='*', color='pink')
        plt.plot(timesteps, worms, label="Worms", linestyle='-', marker='v', color='green')

        plt.xlabel("Time Step")
        plt.ylabel("Counts")
        plt.title("Insects, Birds, Bats Counts Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 创建第二个窗口：显示农作物能量收获率
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, harvest_energy_ratio, label="Harvest Energy Ratio", linestyle='-', marker='x', color='red')

        plt.xlabel("Time Step")
        plt.ylabel("Harvest Energy Ratio")
        plt.title("Crop Harvest Energy Ratio Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 创建第三个窗口：显示人类活动区平均土地肥力
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, avg_human_activity_fertility, label="Avg Human Activity Fertility", linestyle='-', marker='o', color='green')

        plt.xlabel("Time Step")
        plt.ylabel("Average Fertility")
        plt.title("Average Fertility in Human Activity Area Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def parse_save_files(self):
        """解析保存文件夹中的所有文件，并按时间排序"""
        files = [
            f for f in os.listdir(self.save_folder) 
            if f.startswith("sandbox_state_") and f.endswith(".json")
        ]
        self.history_files = sorted(
            [os.path.join(self.save_folder, f) for f in files],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.current_file_index = len(self.history_files) - 1

    def save_state(self):
        # 按时间戳命名文件
        timestamp = int(time.time())
        file_name = f"sandbox_state_{timestamp}.json"
        file_path = os.path.join("save_history", file_name)

        # 创建文件夹（如果不存在）
        os.makedirs("save_history", exist_ok=True)

        # 保存沙盒状态
        state = {
            "layers": {k: v.tolist() for k, v in self.layers.items()},
            "timers": {
                "crop_timers": self.crop_timers.tolist(),
                "herbicide_timers": self.herbicide_timers.tolist(),
                "insecticide_timers": self.insecticide_timers.tolist(),
                "tree_timers": self.tree_timers.tolist()  # 保存树木的计时器
            },
            "energies": {
                "crop_energies": self.crop_energies.tolist(),
                "insect_energies": self.insect_energies.tolist(),
                "bird_energies": self.bird_energies.tolist(),
                "bat_energies": self.bat_energies.tolist(),
                "tree_energies": self.tree_energies.tolist()  # 保存树木的能量信息
            },
            "season": self.season,
            "time_step": self.current_time_step
        }

        with open(file_path, "w") as f:
            json.dump(state, f)

        # 更新保存历史
        if not hasattr(self, "save_history"):
            self.save_history = []
        self.save_history.append(file_path)

        print(f"Sandbox state saved: {file_path}")

    def load_state(self, file_path):
        with open(file_path, "r") as f:
            state = json.load(f)

        # 恢复状态
        for layer_name, layer_data in state["layers"].items():
            self.layers[layer_name] = np.array(layer_data, dtype=int)

        # 恢复计时器
        self.crop_timers = np.array(state["timers"]["crop_timers"], dtype=int)
        self.herbicide_timers = np.array(state["timers"]["herbicide_timers"], dtype=int)
        self.insecticide_timers = np.array(state["timers"]["insecticide_timers"], dtype=int)
        self.tree_timers = np.array(state["timers"]["tree_timers"], dtype=int)  # 恢复树木计时器

        # 恢复能量
        self.crop_energies = np.array(state["energies"]["crop_energies"], dtype=int)
        self.insect_energies = np.array(state["energies"]["insect_energies"], dtype=int)
        self.bird_energies = np.array(state["energies"]["bird_energies"], dtype=int)
        self.bat_energies = np.array(state["energies"]["bat_energies"], dtype=int)
        self.tree_energies = np.array(state["energies"]["tree_energies"], dtype=int)  # 恢复树木能量

        self.season = state["season"]
        self.current_time_step = state["time_step"]

        print(f"State loaded from {file_path}")

    def load_next_file(self):
        """加载下一个保存文件"""
        if not self.history_files:
            print("No saved files available.")
            return

        # 循环到下一个文件
        self.current_file_index = (self.current_file_index + 1) % len(self.history_files)
        file_to_load = self.history_files[self.current_file_index]
        self.load_state(file_to_load)
        print(f"Loaded sandbox state: {file_to_load}")

    def load_previous_file(self):
        """加载上一个保存文件"""
        if not self.history_files:
            print("No saved files available.")
            return

        # 循环到上一个文件
        self.current_file_index = (self.current_file_index - 1) % len(self.history_files)
        file_to_load = self.history_files[self.current_file_index]
        self.load_state(file_to_load)
        print(f"Loaded sandbox state: {file_to_load}")

    def render(self):
        """
        Render the sandbox with pygame.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.cell_size + 300, self.height * self.cell_size + 150))
        pygame.display.set_caption("Interactive Sandbox")
        clock = pygame.time.Clock()  # Control frame rate
        FPS = 30

        font = pygame.font.SysFont(None, 24)

        running = True
        while running:
            screen.fill((0, 0, 0))  # Clear the screen

            # Draw legend (only calculated once)
            legend_x = self.width * self.cell_size + 20
            legend_y = 20
            legend_items = [
                ("Land", self.colors["Land"]),
                ("Seedling", self.colors["Seedling"]),
                ("Tree Trunk", self.colors["TreeTrunk"]),
                ("Leaves", self.colors["MatureTree"]),
                ("Crop", self.colors["Crop"]),
                ("Insect", self.colors["Insect"]),  # 添加昆虫图例
                ("Bird", self.colors["Bird"]),
                ("Bat", self.colors["Bat"]),
                ("Bee", self.colors["Bee"]),
                ("Worm", self.colors["Worm"]),
                ("Herbicide", (255, 0, 0, 128)), # 半透明红色
                ("Insecticide", (0, 191, 255, 128))  # 半透明天蓝色
            ]

            for name, color in legend_items:
                pygame.draw.rect(screen, color, (legend_x, legend_y, 20, 20))
                text_surface = font.render(name, True, (255, 255, 255))
                screen.blit(text_surface, (legend_x + 30, legend_y))
                legend_y += 30

            # Pre-calculate all layers to draw
            land_layer = self.layer_visibility["Land"]
            seedlings_layer = self.layer_visibility["Seedlings"]
            mature_trees_layer = self.layer_visibility["MatureTrees"]
            crops_layer = self.layer_visibility["Crops"]
            insect_layer = self.layer_visibility["Insects"]
            bird_layer = self.layer_visibility["Birds"]
            bat_layer = self.layer_visibility["Bats"]
            bee_layer = self.layer_visibility["Bees"]
            worm_layer = self.layer_visibility["Worms"]
            herbicide_layer = self.layer_visibility["Herbicide"]
            insecticide_layer = self.layer_visibility["Insecticide"]

            # Render layers efficiently
            for y in range(self.height):
                for x in range(self.width):
                    # Land layer
                    if land_layer:
                        fertility = self.layers["Land"][y, x]
                        # 假设最大肥力为 10000（或从配置文件中读取最大值）
                        max_fertility = self.config["land_default_fertility"]  # 使用配置中的最大值，默认 10000
                        normalized_fertility = min(fertility / max_fertility, 1)  # 确保归一化范围在 [0, 1]
                        land_color = tuple(int(c * normalized_fertility) for c in self.colors["Land"])  # 根据归一化肥力计算颜色
                        pygame.draw.rect(
                            screen, land_color,
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

                    # Seedlings layer
                    if seedlings_layer and self.layers["Seedlings"][y, x] == 1:
                        pygame.draw.rect(
                            screen, self.colors["Seedling"],
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

                    # Mature Trees layer
                    if mature_trees_layer:
                        if self.layers["MatureTrees"][y, x] == 1:  # Tree trunk
                            pygame.draw.rect(
                                screen, self.colors["TreeTrunk"],
                                pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                            )
                        elif self.layers["MatureTrees"][y, x] == 2:  # Tree leaf
                            pygame.draw.rect(
                                screen, self.colors["MatureTree"],
                                pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                            )

                    # 农作物显示逻辑
                    if crops_layer and self.layers["Crops"][y, x] == 1:
                        crop_energy = self.crop_energies[y, x]
                        # 根据能量动态调整颜色
                        normalized_energy = max(0, min(crop_energy / self.config["crop_max_energy"], 1))
                        crop_color = (
                            int(100 + 155 * normalized_energy),  # 从灰色到黄色
                            int(100 + 155 * normalized_energy),
                            100
                        )
                        pygame.draw.rect(
                            screen, crop_color,
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

                    # 渲染昆虫图层
                    if insect_layer:
                        # Ensure insects are rendered correctly in each cell
                        if self.layers["Insects"][y, x] == 1:
                            pygame.draw.circle(
                                screen, self.colors["Insect"],
                                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                                self.cell_size // 3
                            )

                    # 渲染鸟图层
                    if bird_layer:
                        # Ensure birds are rendered correctly in each cell
                        if self.layers["Birds"][y, x] == 1:
                            pygame.draw.circle(
                                screen, self.colors["Bird"],
                                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                                self.cell_size // 2
                            )

                    # 渲染蝙蝠图层
                    if bat_layer:
                        if self.layers["Bats"][y, x] == 1:
                            pygame.draw.circle(
                                screen, self.colors["Bat"],
                                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                                self.cell_size // 2
                            )

                    if bee_layer:
                        if self.layers["Bees"][y, x] == 1:
                            pygame.draw.circle(
                                screen, self.colors["Bee"],
                                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                                self.cell_size // 3
                            )

                    if worm_layer:
                        if self.layers["Worms"][y, x] == 1:
                            pygame.draw.circle(
                                screen, self.colors["Worm"],
                                (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                                self.cell_size // 3
                            )

                    # 渲染除草剂图层
                    if herbicide_layer:
                        if self.layers["Herbicide"][y, x] == 1:
                            herbicide_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                            herbicide_surface.fill((255, 0, 0, 128))  # 半透明红色
                            screen.blit(herbicide_surface, (x * self.cell_size, y * self.cell_size))

                    # 渲染杀虫剂图层
                    if insecticide_layer:
                        if self.layers["Insecticide"][y, x] == 1:
                            insecticide_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                            insecticide_surface.fill((0, 191, 255, 128))  # 半透明天蓝色
                            screen.blit(insecticide_surface, (x * self.cell_size, y * self.cell_size))

            # Draw toolbar
            toolbar_y = self.height * self.cell_size
            pygame.draw.rect(screen, (50, 50, 50), (0, toolbar_y, self.width * self.cell_size + 300, 50))

            # Add brush buttons
            seedling_button = pygame.Rect(20, toolbar_y + 10, 80, 30)
            mature_tree_button = pygame.Rect(120, toolbar_y + 10, 80, 30)
            crop_button = pygame.Rect(220, toolbar_y + 10, 80, 30)
            insect_button = pygame.Rect(320, toolbar_y + 10, 80, 30)
            bird_button = pygame.Rect(420, toolbar_y + 10, 80, 30)
            bat_button = pygame.Rect(520, toolbar_y + 10, 80, 30)
            bee_button = pygame.Rect(620, toolbar_y + 10, 80, 30)
            worm_button = pygame.Rect(720, toolbar_y + 10, 80, 30)
            herbicide_button = pygame.Rect(820, toolbar_y + 10, 80, 30)
            insecticide_button = pygame.Rect(920, toolbar_y + 10, 80, 30)
            delete_button = pygame.Rect(1020, toolbar_y + 10, 80, 30)

            pygame.draw.rect(screen, (0, 255, 0), seedling_button)
            pygame.draw.rect(screen, (0, 100, 0), mature_tree_button)
            pygame.draw.rect(screen, (255, 165, 0), crop_button)
            pygame.draw.rect(screen, (255, 255, 0), insect_button)  # 黄色表示昆虫
            pygame.draw.rect(screen, (0, 0, 255), bird_button)  # 蓝色表示鸟单位
            pygame.draw.rect(screen, (255, 0, 255), bat_button)  # 紫色表示蝙蝠单位
            pygame.draw.rect(screen, (255, 192, 203), bee_button)  # 粉色按钮表示蜜蜂单位
            pygame.draw.rect(screen, (128, 128, 128), worm_button)
            pygame.draw.rect(screen, (255, 0, 0), herbicide_button)  # 红色按钮
            pygame.draw.rect(screen, (0, 191, 255), insecticide_button)  # 天蓝色按钮
            pygame.draw.rect(screen, (255, 0, 0), delete_button)

            screen.blit(font.render("Seedling", True, (0, 0, 0)), (25, toolbar_y + 15))
            screen.blit(font.render("Mature", True, (0, 0, 0)), (125, toolbar_y + 15))
            screen.blit(font.render("Crop", True, (0, 0, 0)), (225, toolbar_y + 15))
            screen.blit(font.render("Insect", True, (0, 0, 0)), (325, toolbar_y + 15))
            screen.blit(font.render("Bird", True, (0, 0, 0)), (425, toolbar_y + 15))
            screen.blit(font.render("Bat", True, (0, 0, 0)), (525, toolbar_y + 15))
            screen.blit(font.render("Bee", True, (0, 0, 0)), (625, toolbar_y + 15))
            screen.blit(font.render("Worm", True, (0, 0, 0)), (725, toolbar_y + 15))
            screen.blit(font.render("PES-her", True, (0, 0, 0)), (825, toolbar_y + 15))
            screen.blit(font.render("PES-ins", True, (0, 0, 0)), (925, toolbar_y + 15))
            screen.blit(font.render("Delete", True, (0, 0, 0)), (1025, toolbar_y + 15))

            # Handle events
            mouse_pos = pygame.mouse.get_pos()
            grid_x = mouse_pos[0] // self.cell_size
            grid_y = mouse_pos[1] // self.cell_size

            # Adjust information bar height dynamically
            info_bar_y = self.height * self.cell_size + 50
            info_bar_height = 200  # Set a larger height for the information bar
            pygame.draw.rect(screen, (0, 0, 0), (0, info_bar_y, self.width * self.cell_size + 300, info_bar_height))

            # Initialize line offset for dynamic positioning of text
            line_offset = 5  # Start with an initial offset from the top of the info bar

            # Display current time step (highest priority, display at the very top)
            time_step_text = f"Time Step: {self.current_time_step}"
            time_step_surface = font.render(time_step_text, True, (255, 255, 255))
            screen.blit(time_step_surface, (10, info_bar_y + line_offset))
            line_offset += 20  # Increment line offset for the next line

            # Display pixel information (below the harvest information)
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                fertility = self.layers["Land"][grid_y, grid_x]
                pixel_info = f"Fertility: {fertility}"

                if self.layers["Seedlings"][grid_y, grid_x] == 1:
                    energy = self.tree_energies[grid_y, grid_x]
                    pixel_info += f" | Seedling | Energy: {energy}"
                elif self.layers["MatureTrees"][grid_y, grid_x] == 1:
                    energy = self.tree_energies[grid_y, grid_x]
                    pixel_info += f" | Tree Trunk | Energy: {energy}"
                elif self.layers["MatureTrees"][grid_y, grid_x] == 2:
                    pixel_info += " | Leaf"
                elif self.layers["Crops"][grid_y, grid_x] == 1:
                    crop_energy = self.crop_energies[grid_y, grid_x]
                    pixel_info += f" | Crop | Energy: {crop_energy}"
                elif self.layers["Insects"][grid_y, grid_x] == 1:
                    insect_energy = self.insect_energies[grid_y, grid_x]
                    pixel_info += f" | Insect | Energy: {insect_energy}"
                elif self.layers["Birds"][grid_y, grid_x] == 1:
                    bird_energy = self.bird_energies[grid_y, grid_x]
                    bird_age = self.bird_ages[grid_y, grid_x]
                    pixel_info += f" | Bird | Energy: {bird_energy} | Age: {bird_age}"
                elif self.layers["Bats"][grid_y, grid_x] == 1:
                    bat_energy = self.bat_energies[grid_y, grid_x]
                    bat_age = self.bat_ages[grid_y, grid_x]
                    pixel_info += f" | Bat | Energy: {bat_energy} | Age: {bat_age}"
                elif self.layers["Bees"][grid_y, grid_x] == 1:
                    bee_energy = self.bee_energies[grid_y, grid_x]
                    bee_age = self.bee_ages[grid_y, grid_x]
                    pixel_info += f" | Bee | Energy: {bee_energy} | Age: {bee_age}"
                elif self.layers["Worms"][grid_y, grid_x] == 1:
                    worm_energy = self.worm_energies[grid_y, grid_x]
                    pixel_info += f" | Worm | Energy: {worm_energy}"
                elif self.layers["Herbicide"][grid_y, grid_x] == 1:
                    pixel_info += " | Herbicide Active"
                elif self.layers["Insecticide"][grid_y, grid_x] == 1:
                    pixel_info += " | Insecticide Active"

                # Render pixel info text below the harvest information
                info_text = font.render(pixel_info, True, (255, 255, 255))
                screen.blit(info_text, (10, info_bar_y + line_offset))
                line_offset += 20  # Increment line offset for future expansion

                season_text = f"Current Season: {['Spring', 'Summer', 'Autumn', 'Winter'][self.season]}"
                season_surface = font.render(season_text, True, (255, 255, 255))
                screen.blit(season_surface, (10, info_bar_y + line_offset))
                line_offset += 20

            # Check for mouse events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.current_time_step > 0:
                        self.save_results()
                        self.plot_stats()
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.simulating = not self.simulating
                    elif event.key == pygame.K_1:
                        self.layer_visibility["MatureTrees"] = not self.layer_visibility["MatureTrees"]
                    elif event.key == pygame.K_2:
                        self.layer_visibility["Crops"] = not self.layer_visibility["Crops"]
                    elif event.key == pygame.K_3:
                        self.layer_visibility["Insects"] = not self.layer_visibility["Insects"]
                    elif event.key == pygame.K_4:
                        self.layer_visibility["Birds"] = not self.layer_visibility["Birds"]
                    elif event.key == pygame.K_5:
                        self.layer_visibility["Bats"] = not self.layer_visibility["Bats"]
                    elif event.key == pygame.K_6:
                        self.layer_visibility["Bees"] = not self.layer_visibility["Bees"]
                    elif event.key == pygame.K_7:
                        self.layer_visibility["Worms"] = not self.layer_visibility["Worms"]
                    elif event.key == pygame.K_8:
                        self.layer_visibility["Herbicide"] = not self.layer_visibility["Herbicide"]
                    elif event.key == pygame.K_9:
                        self.layer_visibility["Insecticide"] = not self.layer_visibility["Insecticide"]
                    elif event.key == pygame.K_r:  # 重置
                        self.layer_visibility["MatureTrees"] = True
                        self.layer_visibility["Crops"] = True
                        self.layer_visibility["Insects"] = True
                        self.layer_visibility["Birds"] = True
                        self.layer_visibility["Bats"] = True
                        self.layer_visibility["Bees"] = True
                        self.layer_visibility["Worms"] = True
                        self.layer_visibility["Herbicide"] = True
                        self.layer_visibility["Insecticide"] = True
                    elif event.key == pygame.K_0:  # 读取状态
                        self.layer_visibility["MatureTrees"] = False
                        self.layer_visibility["Crops"] = False
                        self.layer_visibility["Insects"] = False
                        self.layer_visibility["Birds"] = False
                        self.layer_visibility["Bats"] = False
                        self.layer_visibility["Bees"] = False
                        self.layer_visibility["Worms"] = False
                        self.layer_visibility["Herbicide"] = False
                        self.layer_visibility["Insecticide"] = False
                    elif event.key == pygame.K_s:  # 保存当前状态
                        self.save_state()
                    elif event.key == pygame.K_n:  # 上一个文件
                        self.load_previous_file()
                    elif event.key == pygame.K_m:  # 下一个文件
                        self.load_next_file()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if toolbar_y <= mouse_pos[1] < toolbar_y + 50:
                        if seedling_button.collidepoint(mouse_pos):
                            self.brush = "Seedling"
                        elif mature_tree_button.collidepoint(mouse_pos):
                            self.brush = "MatureTree"
                        elif crop_button.collidepoint(mouse_pos):
                            self.brush = "Crop"
                        elif insect_button.collidepoint(mouse_pos):
                            self.brush = "Insect"
                        elif bird_button.collidepoint(mouse_pos):
                            self.brush = "Bird"
                        elif bat_button.collidepoint(mouse_pos):
                            self.brush = "Bat"
                        elif bee_button.collidepoint(mouse_pos):
                            self.brush = "Bee"
                        elif worm_button.collidepoint(mouse_pos):
                            self.brush = "Worm"
                        elif herbicide_button.collidepoint(mouse_pos):
                            self.brush = "Herbicide"
                        elif insecticide_button.collidepoint(mouse_pos):
                            self.brush = "Insecticide"
                        elif delete_button.collidepoint(mouse_pos):
                            self.brush = "Delete"
                elif event.type == pygame.MOUSEMOTION:
                    # Check if mouse button is pressed and dragging
                    if pygame.mouse.get_pressed()[0]:  # Left mouse button
                        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                            if self.brush == "Seedling":
                                self.layers["Seedlings"][grid_y, grid_x] = 1
                                self.tree_energies[grid_y, grid_x] = self.config["seedling_initial_energy"]
                            elif self.brush == "MatureTree":
                                self.layers["MatureTrees"][grid_y, grid_x] = 1
                                self.tree_energies[grid_y, grid_x] = self.config["mature_tree_initial_energy"]
                                # Add leaves around trunk
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < self.width and 0 <= ny < self.height and self.layers["MatureTrees"][ny, nx] == 0:
                                        self.layers["MatureTrees"][ny, nx] = 2  # Mark as leaf
                            elif self.brush == "Crop":
                                self.layers["Seedlings"][grid_y, grid_x] = 0
                                self.layers["MatureTrees"][grid_y, grid_x] = 0
                                # Clean unconnected leaves around the deleted cell
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < self.width and 0 <= ny < self.height:
                                        if self.layers["MatureTrees"][ny, nx] == 2:  # If it's a leaf
                                            self.clean_unconnected_leaves(nx, ny)
                                self.layers["Crops"][grid_y, grid_x] = 1
                                self.crop_energies[grid_y, grid_x] = self.config["crop_initial_energy"]
                                self.crop_timers[grid_y, grid_x] = 0
                            elif self.brush == "Insect":
                                self.layers["Insects"][grid_y, grid_x] = 1
                                self.insect_energies[grid_y, grid_x] = self.config["insect_initial_energy"]
                            elif self.brush == "Bird":
                                self.layers["Birds"][grid_y, grid_x] = 1
                                self.bird_energies[grid_y, grid_x] = self.config["bird_initial_energy"]
                                self.bird_timers[grid_y, grid_x] = 0
                            elif self.brush == "Bat":
                                self.layers["Bats"][grid_y, grid_x] = 1
                                self.bat_energies[grid_y, grid_x] = self.config["bat_initial_energy"]
                                self.bat_ages[grid_y, grid_x] = 0
                            elif self.brush == "Bee":
                                self.layers["Bees"][grid_y, grid_x] = 1
                                self.bee_energies[grid_y, grid_x] = self.config["bee_initial_energy"]
                                self.bee_ages[grid_y, grid_x] = 0
                            elif self.brush == "Worm":
                                self.layers["Worms"][grid_y, grid_x] = 1
                                self.worm_energies[grid_y, grid_x] = self.config["worm_initial_energy"]
                            elif self.brush == "Herbicide":
                                self.layers["Herbicide"][grid_y, grid_x] = 1
                                self.herbicide_timers[grid_y, grid_x] = 0
                            elif self.brush == "Insecticide":
                                self.layers["Insecticide"][grid_y, grid_x] = 1
                                self.insecticide_timers[grid_y, grid_x] = 0
                            elif self.brush == "Delete":
                                self.layers["Seedlings"][grid_y, grid_x] = 0
                                self.layers["MatureTrees"][grid_y, grid_x] = 0
                                self.layers["Crops"][grid_y, grid_x] = 0
                                self.layers["Insects"][grid_y, grid_x] = 0
                                self.layers["Birds"][grid_y, grid_x] = 0
                                self.layers["Bats"][grid_y, grid_x] = 0
                                self.layers["Bees"][grid_y, grid_x] = 0
                                self.layers["Worms"][grid_y, grid_x] = 0
                                self.layers["Herbicide"][grid_y, grid_x] = 0
                                self.layers["Insecticide"][grid_y, grid_x] = 0
                                # Clean unconnected leaves around the deleted cell
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < self.width and 0 <= ny < self.height:
                                        if self.layers["MatureTrees"][ny, nx] == 2:  # If it's a leaf
                                            self.clean_unconnected_leaves(nx, ny)

            # Simulate steps if active
            if self.simulating:
                self.simulate_step()

            pygame.display.flip()
            clock.tick(FPS)

# Example usage
if __name__ == "__main__":
    sandbox = Sandbox(config_file="config_eworm15_add.json")
    sandbox.render()
