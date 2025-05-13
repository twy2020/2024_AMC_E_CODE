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
        self.colors = {
            "Land": tuple(self.config["land_color"]),
            "Seedling": (0, 255, 0),  # Green for seedlings
            "MatureTree": (0, 100, 0),  # Dark green for mature trees
            "TreeTrunk": (255, 165, 0),  # Orange for tree trunk
        }
        self.layer_visibility = {"Land": True, "Seedlings": True, "MatureTrees": True}

        # Brush state
        self.brush = "Seedling"  # Default brush

    def clean_unconnected_leaves(self, start_x, start_y):
        """
        Remove leaves that are not directly connected to any tree trunk.
        """
        visited = np.zeros_like(self.layers["MatureTrees"], dtype=bool)

        def is_directly_connected_to_trunk(x, y):
            """
            Check if a leaf is directly connected to a tree trunk.
            """
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.layers["MatureTrees"][ny, nx] == 1:  # Tree trunk
                        return True
            return False

        def dfs(x, y):
            if not (0 <= x < self.width and 0 <= y < self.height):
                return False
            if visited[y, x]:
                return False
            visited[y, x] = True

            if self.layers["MatureTrees"][y, x] == 2:  # Leaf
                # If not directly connected to a trunk, mark for removal
                if not is_directly_connected_to_trunk(x, y):
                    self.layers["MatureTrees"][y, x] = 0

                # Continue checking neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(x + dx, y + dy)

        # Start DFS from the given position
        dfs(start_x, start_y)

    def render(self):
        """
        Render the sandbox with pygame.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.cell_size + 200, self.height * self.cell_size + 50))
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

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                fertility = self.layers["Land"][grid_y, grid_x]
                info_text = font.render(f"Fertility: {fertility}", True, (255, 255, 255))
                screen.blit(info_text, (self.width * self.cell_size + 20, 20))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if toolbar_y <= mouse_pos[1] < toolbar_y + 50:
                        # Handle toolbar button clicks
                        if seedling_button.collidepoint(mouse_pos):
                            self.brush = "Seedling"
                        elif mature_tree_button.collidepoint(mouse_pos):
                            self.brush = "MatureTree"
                        elif delete_button.collidepoint(mouse_pos):
                            self.brush = "Delete"
                    else:
                        # Handle cell interactions
                        if self.brush == "Seedling":
                            self.layers["Seedlings"][grid_y, grid_x] = 1
                            self.layers["MatureTrees"][grid_y, grid_x] = 0
                        elif self.brush == "MatureTree":
                            # Place tree trunk at the center
                            self.layers["Seedlings"][grid_y, grid_x] = 0
                            self.layers["MatureTrees"][grid_y, grid_x] = 1
                            # Place tree leaves around the trunk
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    if self.layers["Seedlings"][ny, nx] == 0 and self.layers["MatureTrees"][ny, nx] == 0:
                                        self.layers["MatureTrees"][ny, nx] = 2  # Mark as leaf
                        elif self.brush == "Delete":
                            self.layers["Seedlings"][grid_y, grid_x] = 0
                            self.layers["MatureTrees"][grid_y, grid_x] = 0
                            # Check and clean nearby leaves
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.width and 0 <= ny < self.height:
                                    if self.layers["MatureTrees"][ny, nx] == 2:  # If it's a leaf
                                        self.clean_unconnected_leaves(nx, ny)

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

# Example usage
if __name__ == "__main__":
    sandbox = Sandbox(config_file="config_tree_test.json")
    sandbox.render()