import pygame
import numpy as np
import json
import os

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

        self.width = self.config["sandbox_width"]
        self.height = self.config["sandbox_height"]
        self.cell_size = self.config["cell_size"]

        # Initialize fixed fertility for the land layer
        self.layers = {
            "Land": np.full((self.height, self.width), 100, dtype=int),  # Fixed fertility at 100
            "Seedlings": np.zeros((self.height, self.width), dtype=int),
            "MatureTrees": np.zeros((self.height, self.width), dtype=int),
        }
        self.tree_timers = np.zeros((self.height, self.width), dtype=int)  # Track growth timers for seedlings
        self.tree_energies = np.full((self.height, self.width), -1, dtype=int)  # Track energy for trees

        self.colors = {
            "Land": tuple(self.config["land_color"]),
            "Seedling": (0, 255, 0),  # Green for seedlings
            "MatureTree": (0, 100, 0),  # Dark green for mature trees
            "TreeTrunk": (255, 165, 0),  # Orange for tree trunk
        }
        self.layer_visibility = {"Land": True, "Seedlings": True, "MatureTrees": True}

        # Brush state
        self.brush = "Seedling"  # Default brush

        # Simulation state
        self.simulating = False
        self.current_time_step = 0

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
        seedling_energy_loss_no_fertility = self.config.get("seedling_energy_loss_no_fertility", 5)
        mature_tree_energy_loss_no_fertility = self.config.get("mature_tree_energy_loss_no_fertility", 15)

        # Update Seedlings
        for y in range(self.height):
            for x in range(self.width):
                if self.layers["Seedlings"][y, x] == 1:
                    self.tree_timers[y, x] += 1
                    self.layers["Land"][y, x] = max(0, self.layers["Land"][y, x] - self.config["seedling_fertility_cost"])

                    if self.layers["Land"][y, x] == 0:
                        # Reduce energy when no fertility is available
                        self.tree_energies[y, x] -= seedling_energy_loss_no_fertility
                    else:
                        # Gain energy when fertility is available
                        self.tree_energies[y, x] = min(
                            self.tree_energies[y, x] + self.config["seedling_energy_gain"],
                            self.config["seedling_max_energy"]
                        )

                    # Check if the seedling should die
                    if self.tree_energies[y, x] < self.config["seedling_min_energy"]:
                        self.layers["MatureTrees"][y, x] = 0
                        self.layers["Land"][y, x] = min(
                            100,  # 限制肥力最大值为 100
                            self.layers["Land"][y, x] + int(self.tree_energies[y, x] * self.config["fertility_return_ratio"])
                        )
                        self.tree_energies[y, x] = -1

                    # Check if the seedling should mature
                    elif self.tree_timers[y, x] >= self.config["seedling_growth_time"]:
                        self.layers["Seedlings"][y, x] = 0
                        self.layers["MatureTrees"][y, x] = 1
                        self.tree_energies[y, x] = self.config["mature_tree_initial_energy"]

        # Update Mature Trees
        for y in range(self.height):
            for x in range(self.width):
                if self.layers["MatureTrees"][y, x] == 1:
                    self.layers["Land"][y, x] = max(0, self.layers["Land"][y, x] - self.config["mature_tree_fertility_cost"])

                    if self.layers["Land"][y, x] == 0:
                        # Reduce energy when no fertility is available
                        self.tree_energies[y, x] -= mature_tree_energy_loss_no_fertility
                    else:
                        # Gain energy when fertility is available
                        self.tree_energies[y, x] = min(
                            self.tree_energies[y, x] + self.config["mature_tree_energy_gain"],
                            self.config["mature_tree_max_energy"]
                        )

                    # Add tree leaves around trunk
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.layers["MatureTrees"][ny, nx] == 0:
                                self.layers["MatureTrees"][ny, nx] = 2

                    # When tree trunk energy is below minimum, remove trunk and clean leaves
                    if self.tree_energies[y, x] < self.config["mature_tree_min_energy"]:
                        # Remove trunk
                        self.layers["MatureTrees"][y, x] = 0
                        self.layers["Land"][y, x] = min(
                            100,  # 限制肥力最大值为 100
                            self.layers["Land"][y, x] + int(self.tree_energies[y, x] * self.config["fertility_return_ratio"])
                        )
                        self.tree_energies[y, x] = -1
                        
                        # Clean unconnected leaves around the deleted cell
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if self.layers["MatureTrees"][ny, nx] == 2:  # If it's a leaf
                                    self.clean_unconnected_leaves(nx, ny)

                    # Reproduction logic (generate one seedling in a 5x5 area)
                    elif np.random.rand() < self.config["mature_tree_reproduction_chance"]:
                        # Randomly pick a position in a 5x5 area around the trunk
                        potential_positions = [
                            (x + dx, y + dy)
                            for dx in range(-2, 3)  # -2 to 2 inclusive
                            for dy in range(-2, 3)
                            if (dx != 0 or dy != 0) and 0 <= x + dx < self.width and 0 <= y + dy < self.height
                        ]
                        if potential_positions:
                            nx, ny = potential_positions[np.random.randint(len(potential_positions))]
                            # Check if the selected cell is not occupied by a trunk or leaf
                            if self.layers["MatureTrees"][ny, nx] == 0 and self.layers["Seedlings"][ny, nx] == 0:
                                self.layers["Seedlings"][ny, nx] = 1
                                self.tree_timers[ny, nx] = 0
                                self.tree_energies[ny, nx] = self.config["seedling_initial_energy"]

        self.current_time_step += 1

    def render(self):
        """
        Render the sandbox with pygame.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.cell_size + 200, self.height * self.cell_size + 100))
        pygame.display.set_caption("Interactive Sandbox")
        clock = pygame.time.Clock()  # Control frame rate
        FPS = 10

        font = pygame.font.SysFont(None, 24)

        running = True
        while running:
            screen.fill((0, 0, 0))

            # Draw legend
            legend_x = self.width * self.cell_size + 20
            legend_y = 20
            font = pygame.font.SysFont(None, 24)

            # Define legend items
            legend_items = [
                ("Land", self.colors["Land"]),
                ("Seedling", self.colors["Seedling"]),
                ("Tree Trunk", self.colors["TreeTrunk"]),
                ("Leaves", self.colors["MatureTree"])
            ]

            for name, color in legend_items:
                pygame.draw.rect(screen, color, (legend_x, legend_y, 20, 20))
                text_surface = font.render(name, True, (255, 255, 255))
                screen.blit(text_surface, (legend_x + 30, legend_y))
                legend_y += 30

            # Draw cells
            for y in range(self.height):
                for x in range(self.width):
                    # Draw Land layer
                    if self.layer_visibility["Land"]:
                        fertility = self.layers["Land"][y, x]
                        land_color = tuple(int(c * fertility / 100) for c in self.colors["Land"])
                        pygame.draw.rect(
                            screen, land_color,
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

                    # Draw seedlings
                    if self.layer_visibility["Seedlings"] and self.layers["Seedlings"][y, x] == 1:
                        pygame.draw.rect(
                            screen, self.colors["Seedling"],
                            pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        )

                    # Draw mature trees
                    if self.layer_visibility["MatureTrees"]:
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

            # Draw toolbar
            toolbar_y = self.height * self.cell_size
            pygame.draw.rect(screen, (50, 50, 50), (0, toolbar_y, self.width * self.cell_size + 200, 50))

            # Add brush buttons
            seedling_button = pygame.Rect(20, toolbar_y + 10, 80, 30)
            mature_tree_button = pygame.Rect(120, toolbar_y + 10, 80, 30)
            delete_button = pygame.Rect(220, toolbar_y + 10, 80, 30)

            pygame.draw.rect(screen, (0, 255, 0), seedling_button)
            pygame.draw.rect(screen, (0, 100, 0), mature_tree_button)
            pygame.draw.rect(screen, (255, 0, 0), delete_button)

            screen.blit(font.render("Seedling", True, (0, 0, 0)), (25, toolbar_y + 15))
            screen.blit(font.render("Mature", True, (0, 0, 0)), (125, toolbar_y + 15))
            screen.blit(font.render("Delete", True, (0, 0, 0)), (225, toolbar_y + 15))

            # Handle events
            mouse_pos = pygame.mouse.get_pos()
            grid_x = mouse_pos[0] // self.cell_size
            grid_y = mouse_pos[1] // self.cell_size

            # Display pixel information and time step at the bottom
            info_bar_y = self.height * self.cell_size + 50
            pygame.draw.rect(screen, (0, 0, 0), (0, info_bar_y, self.width * self.cell_size + 200, 50))

            # Display current time step
            time_step_text = f"Time Step: {self.current_time_step}"
            time_step_surface = font.render(time_step_text, True, (255, 255, 255))
            screen.blit(time_step_surface, (10, info_bar_y + 5))

            # Display pixel information
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                fertility = self.layers["Land"][grid_y, grid_x]
                pixel_info = f"Fertility: {fertility}"

                # Check for seedlings and mature trees
                if self.layers["Seedlings"][grid_y, grid_x] == 1:
                    energy = self.tree_energies[grid_y, grid_x]
                    pixel_info += f" | Seedling | Energy: {energy}"
                elif self.layers["MatureTrees"][grid_y, grid_x] == 1:
                    energy = self.tree_energies[grid_y, grid_x]
                    pixel_info += f" | Tree Trunk | Energy: {energy}"
                elif self.layers["MatureTrees"][grid_y, grid_x] == 2:
                    pixel_info += " | Leaf"

                info_text = font.render(pixel_info, True, (255, 255, 255))
                screen.blit(info_text, (10, info_bar_y + 25))


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
                        elif delete_button.collidepoint(mouse_pos):
                            self.brush = "Delete"
                    elif 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        if self.brush == "Seedling":
                            self.layers["Seedlings"][grid_y, grid_x] = 1
                            self.tree_energies[grid_y, grid_x] = self.config["seedling_initial_energy"]
                        elif self.brush == "MatureTree":
                            self.layers["MatureTrees"][grid_y, grid_x] = 1
                            self.tree_energies[grid_y, grid_x] = self.config["mature_tree_initial_energy"]
                            # 在树干周围添加树叶
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height and self.layers["MatureTrees"][ny, nx] == 0:
                                    self.layers["MatureTrees"][ny, nx] = 2  # 标记为树叶

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

        pygame.quit()

# Example usage
if __name__ == "__main__":
    sandbox = Sandbox(config_file="config_tree3_mul.json")
    sandbox.render()
