import pygame
import random
import sys, os
import time

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balloon Popping Game")

# Load assets
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BALLOON_IMAGES = []
balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png"]
for file in balloon_files:
    path = os.path.join(base_dir, "assets", file)
    if not os.path.exists(path):
        print(f"Warning: Balloon image {path} not found. Skipping.")
        continue
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (320, 360))
        BALLOON_IMAGES.append(img)
    except pygame.error as e:
        print(f"Warning: Failed to load {path}: {e}. Skipping.")
if not BALLOON_IMAGES:
    print("Error: No valid balloon images loaded. Exiting.")
    pygame.quit()
    sys.exit(1)

# Load sounds
def load_sound(filename):
    path = os.path.join(base_dir, "assets", filename)
    if os.path.exists(path):
        try:
            return pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Failed to load sound {filename}: {e}")
    return None

pop_sound = load_sound("pop.wav")

try:
    boom_path = os.path.join(base_dir, "assets", 'boom1.png')
    boom_img = pygame.image.load(boom_path).convert_alpha()
    boom_img = pygame.transform.scale(boom_img, (290, 180))
except pygame.error as e:
    print(f"Warning: Failed to load crack.png: {e}. No crack image.")
    crack = None

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Clock
clock = pygame.time.Clock()
FPS = 60

# Balloon class
class Balloon:
    def __init__(self):
        self.image = random.choice(BALLOON_IMAGES)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.radius = min(self.width, self.height) // 2
        self.popped = False
        self.reset()

    def reset(self):
        self.x = random.randint(0, WIDTH - self.width)
        self.y = HEIGHT + random.randint(0, 300)
        self.speed = random.uniform(3.75, 6.0)
        self.popped = False

    def update(self):
        if not self.popped:
            self.y -= self.speed
            if self.y + self.height < 0:
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

    def is_clicked(self, pos):
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        dx = pos[0] - cx
        dy = pos[1] - cy
        return dx * dx + dy * dy <= self.radius * self.radius

class CrackEffect:
    def __init__(self, x, y, duration=0.3):
        self.x = x
        self.y = y
        self.start_time = time.time()
        self.duration = duration

    def draw(self, win):
        if time.time() - self.start_time < self.duration and boom_img:
            win.blit(boom_img, (self.x, self.y))
            return True
        return False
    
# Game loop
def main():
    balloons = [Balloon() for _ in range(3)]
    cracks = []
    score = 0
    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        screen.fill(WHITE)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for balloon in balloons[:]:
                    if balloon.is_clicked(pos):
                        balloon.popped = True
                        if pop_sound:
                            pop_sound.play()
                        cracks.append(CrackEffect(balloon.x, balloon.y))
                        balloons.remove(balloon)
                        score += 1

        # Update and draw balloons
        for balloon in balloons:
            balloon.update()
            balloon.draw(screen)

        # Add new balloons if needed
        if len(balloons) < 3:
            balloons.append(Balloon())

        # Draw crack effects
        cracks = [c for c in cracks if c.draw(screen)]

        # Draw score
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # Refresh screen
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
