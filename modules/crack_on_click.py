import pygame
import sys, os

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Click to Crack")

# Load the crack image (should have transparency)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path = os.path.join(base_dir, "assets", "crack.png")
crack_img = pygame.image.load(path).convert_alpha()
crack_img = pygame.transform.scale(crack_img, (260, 140))
crack_rect = crack_img.get_rect()

# Store crack positions
cracks = []

# Main loop
clock = pygame.time.Clock()
running = True

while running:
    screen.fill((255, 255, 255))  # white background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            cracks.append(pos)  # save click position

    # Draw all cracks
    for pos in cracks:
        crack_position = (pos[0] - crack_rect.width // 2, pos[1] - crack_rect.height // 2)
        screen.blit(crack_img, crack_position)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
