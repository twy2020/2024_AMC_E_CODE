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

        # Generate random fertility for the land layer
        self.layers = {
            "Land": np.random.randint(50, 101, size=(self.height, self.width)),  # Fertility between 50 and 100
        }
        self.colors = {
            "Land": tuple(self.config["land_color"]),
        }
        self.layer_visibility = {"Land": True}

    def render(self):
        """
        Render the sandbox with pygame.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.width * self.cell_size + 200, self.height * self.cell_size))
        pygame.display.set_caption("Interactive Sandbox")
        clock = pygame.time.Clock()  # Control frame rate
        FPS = 10

        font = pygame.font.SysFont(None, 24)

        running = True
        while running:
            screen.fill((0, 0, 0))

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

            # Draw legend
            legend_x = self.width * self.cell_size + 20
            legend_y = 20
            pygame.draw.rect(screen, self.colors["Land"], (legend_x, legend_y, 20, 20))
            legend_text = font.render("Land", True, (255, 255, 255))
            screen.blit(legend_text, (legend_x + 30, legend_y))

            # Handle events
            mouse_pos = pygame.mouse.get_pos()
            grid_x = mouse_pos[0] // self.cell_size
            grid_y = mouse_pos[1] // self.cell_size

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                fertility = self.layers["Land"][grid_y, grid_x]
                info_text = font.render(f"Fertility: {fertility}", True, (255, 255, 255))
                screen.blit(info_text, (legend_x, legend_y + 50))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

# Example usage
if __name__ == "__main__":
    sandbox = Sandbox(config_file="config_soil_test.json")
    sandbox.render()
