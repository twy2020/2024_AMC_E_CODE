import pygame
import numpy as np
import json
import os
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
            "Land": np.full((self.height, self.width), 1000, dtype=int),  # Default fertility set to 1000
            "Seedlings": np.zeros((self.height, self.width), dtype=int),  # Seedlings layer
            "MatureTrees": np.zeros((self.height, self.width), dtype=int),  # Mature trees (trunks and leaves)
            "Crops": np.zeros((self.height, self.width), dtype=int)  # Crops layer
        }

        # Initialize state tracking for trees and crops
        self.tree_timers = np.zeros((self.height, self.width), dtype=int)  # Seedling growth timers
        self.tree_energies = np.full((self.height, self.width), -1, dtype=int)  # Tree energy levels
        self.crop_timers = np.zeros((self.height, self.width), dtype=int)  # Crop growth/harvest timers
        self.crop_energies = np.full((self.height, self.width), -1, dtype=int)  # Crop energy levels
        self.crop_harvest_energy = 0  # Accumulated energy from harvested crops

        # Define colors for rendering
        self.colors = {
            "Land": tuple(self.config["land_color"]),  # Convert land color from config
            "Seedling": (0, 255, 0),  # Green for seedlings
            "MatureTree": (0, 100, 0),  # Dark green for mature trees
            "TreeTrunk": (255, 165, 0),  # Orange for tree trunks
            "Crop": (255, 165, 100)  # Orange for crops
        }

        # Layer visibility toggles
        self.layer_visibility = {
            "Land": True,
            "Seedlings": True,
            "MatureTrees": True,
            "Crops": True
        }

        # Brush for user interactions (default to Seedling placement)
        self.brush = "Seedling"

        # Simulation state tracking
        self.simulating = False  # Simulation starts in a paused state
        self.current_time_step = 0  # Time step counter
        self.stats = []  # List to store simulation statistics per time step

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
        crop_energy_loss_no_fertility = self.config.get("crop_energy_loss_no_fertility", 5)
        
        # Seedlings update
        seedling_mask = self.layers["Seedlings"] == 1
        if np.any(seedling_mask):
            # Update timers and fertility
            self.tree_timers[seedling_mask] += 1
            self.layers["Land"][seedling_mask] = np.maximum(0, self.layers["Land"][seedling_mask] - self.config["seedling_fertility_cost"])
            
            # Energy adjustments
            no_fertility_mask = (self.layers["Land"] == 0) & seedling_mask
            with_fertility_mask = (self.layers["Land"] > 0) & seedling_mask
            self.tree_energies[no_fertility_mask] -= seedling_energy_loss_no_fertility
            self.tree_energies[with_fertility_mask] = np.minimum(
                self.tree_energies[with_fertility_mask] + self.config["seedling_energy_gain"],
                self.config["seedling_max_energy"]
            )
            
            # Seedlings dying
            dying_seedling_mask = (self.tree_energies < self.config["seedling_min_energy"]) & seedling_mask
            self.layers["Seedlings"][dying_seedling_mask] = 0
            self.layers["Land"][dying_seedling_mask] = np.minimum(
                1000,
                self.layers["Land"][dying_seedling_mask] + 
                (self.tree_energies[dying_seedling_mask] * self.config["fertility_return_ratio"]).astype(int)
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
            self.layers["Land"][trunk_mask] = np.maximum(0, self.layers["Land"][trunk_mask] - self.config["mature_tree_fertility_cost"])
            no_fertility_trunk = (self.layers["Land"] == 0) & trunk_mask
            with_fertility_trunk = (self.layers["Land"] > 0) & trunk_mask
            self.tree_energies[no_fertility_trunk] -= mature_tree_energy_loss_no_fertility
            self.tree_energies[with_fertility_trunk] = np.minimum(
                self.tree_energies[with_fertility_trunk] + self.config["mature_tree_energy_gain"],
                self.config["mature_tree_max_energy"]
            )
            
            # Handle trunk energy loss and death
            dying_trunk_mask = (self.tree_energies < self.config["mature_tree_min_energy"]) & trunk_mask

            # Remove dying trunks
            self.layers["MatureTrees"][dying_trunk_mask] = 0

            # Add leaves around trunk
            if np.any(trunk_mask):  # 检查是否存在树干
                trunk_positions = np.argwhere(trunk_mask)  # 获取所有树干的位置 (返回 [y, x] 顺序)
                for ty, tx in trunk_positions:  # 注意这里的顺序：先 ty (y)，再 tx (x)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 遍历树干四周的位置
                        nx, ny = tx + dx, ty + dy  # 计算邻居位置（x + dx, y + dy 的顺序）
                        if 0 <= nx < self.width and 0 <= ny < self.height:  # 确保位置合法
                            # 确保该位置没有被其他单元格（种子、农作物等）占用
                            if (
                                self.layers["MatureTrees"][ny, nx] == 0 and  # 不是其他树干或叶子
                                self.layers["Seedlings"][ny, nx] == 0 and   # 不是种子
                                self.layers["Crops"][ny, nx] == 0           # 不是农作物
                            ):
                                self.layers["MatureTrees"][ny, nx] = 2  # 将位置标记为树叶

            # Return fertility to the land
            self.layers["Land"][dying_trunk_mask] = np.minimum(
                1000,
                self.layers["Land"][dying_trunk_mask] + 
                (self.tree_energies[dying_trunk_mask] * self.config["fertility_return_ratio"]).astype(int)
            )

            # Mark tree energies of the dying trunks as invalid
            self.tree_energies[dying_trunk_mask] = -1

            # Clear unconnected leaves
            dying_positions = np.argwhere(dying_trunk_mask)  # Get the positions of the dying trunks
            for dy, dx in dying_positions:
                for dy_offset, dx_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = dy + dy_offset, dx + dx_offset
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # If the adjacent cell is a leaf, check if it's still connected to any trunk
                        if self.layers["MatureTrees"][ny, nx] == 2:  # Leaf
                            self.clean_unconnected_leaves(nx, ny)
            
        # Reproduction logic
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
                    if self.layers["MatureTrees"][ny, nx] == 0 and self.layers["Seedlings"][ny, nx] == 0:
                        # 如果位置为空地，生成种子
                        self.layers["Seedlings"][ny, nx] = 1
                        self.tree_timers[ny, nx] = 0
                        self.tree_energies[ny, nx] = self.config["seedling_initial_energy"]
                    # 如果不是空地，放弃本次生成机会
        
        # Crops update
        crop_mask = self.layers["Crops"] == 1
        if np.any(crop_mask):
            # 增加计时器，用于记录作物生长时间
            self.crop_timers[crop_mask] += 1
            
            # 消耗土地肥力，逐步降低
            self.layers["Land"][crop_mask] = np.maximum(0, self.layers["Land"][crop_mask] - self.config["crop_fertility_cost"])
            
            # 作物能量调整
            no_fertility_crop = (self.layers["Land"] == 0) & crop_mask
            with_fertility_crop = (self.layers["Land"] > 0) & crop_mask
            self.crop_energies[no_fertility_crop] -= crop_energy_loss_no_fertility
            self.crop_energies[with_fertility_crop] = np.minimum(
                self.crop_energies[with_fertility_crop] + self.config["crop_energy_gain"],
                self.config["crop_max_energy"]
            )
            
            # 作物成熟逻辑
            matured_mask = (self.crop_energies >= self.config["crop_max_energy"]) & crop_mask
            harvestable_mask = matured_mask & (self.crop_timers > self.config["crop_maturity_time"] + self.config["crop_harvest_delay"])
            
            # 收割成熟作物并累积能量
            self.crop_harvest_energy += np.sum(self.crop_energies[harvestable_mask])
            self.layers["Crops"][harvestable_mask] = 0  # 清除已收割的作物
            self.crop_energies[harvestable_mask] = -1  # 作物能量置为无效
            
            # 作物死亡逻辑
            dying_crop_mask = crop_mask & (
                (self.crop_energies < self.config["crop_min_energy"]) |  # 能量不足
                (self.crop_timers > self.config["crop_maturity_time"] + self.config["crop_death_delay"])  # 超过死亡延迟
            )
            self.layers["Crops"][dying_crop_mask] = 0  # 清除死亡的作物
            self.layers["Land"][dying_crop_mask] = np.minimum(
                1000,
                self.layers["Land"][dying_crop_mask] + 
                (self.crop_energies[dying_crop_mask] * self.config["crop_fertility_return_ratio_dead"]).astype(int)
            )
            self.crop_energies[dying_crop_mask] = -1  # 作物能量置为无效
            
            # 耕种延迟：死亡或收割后重新种植
            # 获取死亡或收割后的格子
            empty_crop_positions = np.argwhere(self.layers["Crops"] == 0)
            for y, x in empty_crop_positions:
                if self.crop_timers[y, x] >= self.config["crop_plant_delay"]:
                    # 确保格子未被占用
                    if (
                        self.layers["Seedlings"][y, x] == 0 and
                        self.layers["MatureTrees"][y, x] == 0 and
                        self.layers["Crops"][y, x] == 0
                    ):
                        # 重新种植作物
                        self.layers["Crops"][y, x] = 1
                        self.crop_energies[y, x] = self.config["crop_initial_energy"]
                        self.crop_timers[y, x] = 0  # 重置计时器

        
        # Record stats
        seedling_count = np.count_nonzero(self.layers["Seedlings"])
        trunk_count = np.count_nonzero(self.layers["MatureTrees"] == 1)
        leaf_count = np.count_nonzero(self.layers["MatureTrees"] == 2)
        avg_fertility = np.sum(self.layers["Land"]) / (self.width * self.height)
        crop_count = np.count_nonzero(self.layers["Crops"])
        self.stats.append((self.current_time_step, seedling_count, trunk_count, leaf_count, avg_fertility, crop_count, self.crop_harvest_energy))
        
        # Update time step
        self.current_time_step += 1

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = f"./result/simulation_{timestamp}.xlsx"
        config_file = f"./result/simulation_{timestamp}.txt"

        # 保存Excel文件
        df = pd.DataFrame(self.stats, columns=["TimeStep", "Seedlings", "TreeTrunks", "Leaves", "AvgFertility", "Crops", "HarvestEnergy"])
        df.to_excel(excel_file, index=False)

        # 保存配置到文本文件
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"Results saved to {excel_file} and {config_file}.")

    def plot_stats(self):
        # 提取统计数据
        timesteps, seedlings, trunks, leaves, avg_fertility, crops, harvest_energy = zip(*self.stats)

        # 绘制折线图
        plt.figure(figsize=(12, 8))
        plt.plot(timesteps, seedlings, label="Seedlings", linestyle='-', marker='o', color='green')
        plt.plot(timesteps, trunks, label="Tree Trunks", linestyle='-', marker='s', color='brown')
        plt.plot(timesteps, leaves, label="Leaves", linestyle='-', marker='^', color='orange')
        plt.plot(timesteps, avg_fertility, label="Avg Fertility", linestyle='--', marker='x', color='blue')

        plt.xlabel("Time Step")
        plt.ylabel("Count / Avg Fertility")
        plt.title("Simulation Statistics Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def render(self):
        """
        Render the sandbox with pygame.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.cell_size + 200, self.height * self.cell_size + 150))
        pygame.display.set_caption("Interactive Sandbox")
        clock = pygame.time.Clock()  # Control frame rate
        FPS = 100

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
                ("Crop", self.colors["Crop"])
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

            # Render layers efficiently
            for y in range(self.height):
                for x in range(self.width):
                    # Land layer
                    if land_layer:
                        fertility = self.layers["Land"][y, x]
                        # 假设最大肥力为 1000（或从配置文件中读取最大值）
                        max_fertility = self.config.get("max_fertility", 1000)  # 使用配置中的最大值，默认 1000
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

                    # Crops layer
                    if crops_layer and self.layers["Crops"][y, x] == 1:
                        pygame.draw.rect(
                            screen, self.colors["Crop"],
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

            # Draw toolbar
            toolbar_y = self.height * self.cell_size
            pygame.draw.rect(screen, (50, 50, 50), (0, toolbar_y, self.width * self.cell_size + 200, 50))

            # Add brush buttons
            seedling_button = pygame.Rect(20, toolbar_y + 10, 80, 30)
            mature_tree_button = pygame.Rect(120, toolbar_y + 10, 80, 30)
            delete_button = pygame.Rect(220, toolbar_y + 10, 80, 30)
            crop_button = pygame.Rect(320, toolbar_y + 10, 80, 30)

            pygame.draw.rect(screen, (0, 255, 0), seedling_button)
            pygame.draw.rect(screen, (0, 100, 0), mature_tree_button)
            pygame.draw.rect(screen, (255, 0, 0), delete_button)
            pygame.draw.rect(screen, (255, 165, 0), crop_button)

            screen.blit(font.render("Seedling", True, (0, 0, 0)), (25, toolbar_y + 15))
            screen.blit(font.render("Mature", True, (0, 0, 0)), (125, toolbar_y + 15))
            screen.blit(font.render("Delete", True, (0, 0, 0)), (225, toolbar_y + 15))
            screen.blit(font.render("Crop", True, (0, 0, 0)), (325, toolbar_y + 15))

            # Handle events
            mouse_pos = pygame.mouse.get_pos()
            grid_x = mouse_pos[0] // self.cell_size
            grid_y = mouse_pos[1] // self.cell_size

            # Adjust information bar height dynamically
            info_bar_y = self.height * self.cell_size + 50
            info_bar_height = 200  # Set a larger height for the information bar
            pygame.draw.rect(screen, (0, 0, 0), (0, info_bar_y, self.width * self.cell_size + 200, info_bar_height))

            # Initialize line offset for dynamic positioning of text
            line_offset = 5  # Start with an initial offset from the top of the info bar

            # Display current time step (highest priority, display at the very top)
            time_step_text = f"Time Step: {self.current_time_step}"
            time_step_surface = font.render(time_step_text, True, (255, 255, 255))
            screen.blit(time_step_surface, (10, info_bar_y + line_offset))
            line_offset += 20  # Increment line offset for the next line

            # Display total harvested crop energy (below the time step information)
            harvest_info = f"Total Harvested Crop Energy: {self.crop_harvest_energy}"
            harvest_text = font.render(harvest_info, True, (255, 255, 255))
            screen.blit(harvest_text, (10, info_bar_y + line_offset))
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

                # Render pixel info text below the harvest information
                info_text = font.render(pixel_info, True, (255, 255, 255))
                screen.blit(info_text, (10, info_bar_y + line_offset))
                line_offset += 20  # Increment line offset for future expansion

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
                    elif event.key == pygame.K_t:
                        self.layer_visibility["MatureTrees"] = not self.layer_visibility["MatureTrees"]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if toolbar_y <= mouse_pos[1] < toolbar_y + 50:
                        if seedling_button.collidepoint(mouse_pos):
                            self.brush = "Seedling"
                        elif mature_tree_button.collidepoint(mouse_pos):
                            self.brush = "MatureTree"
                        elif crop_button.collidepoint(mouse_pos):
                            self.brush = "Crop"
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
                                self.layers["Crops"][grid_y, grid_x] = 1
                                self.crop_energies[grid_y, grid_x] = self.config["crop_initial_energy"]
                                self.crop_timers[grid_y, grid_x] = 0
                            elif self.brush == "Delete":
                                self.layers["Seedlings"][grid_y, grid_x] = 0
                                self.layers["MatureTrees"][grid_y, grid_x] = 0
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
    sandbox = Sandbox(config_file="config_arg5_add.json")
    sandbox.render()
