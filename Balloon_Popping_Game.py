import pygame
import random
import math
import sys

# --- Init Pygame ---
pygame.init()
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Balloon Pop Game")
CLOCK = pygame.time.Clock()
FPS = 60
pygame.mouse.set_visible(False)

pin_img = pygame.image.load("pin.png")
pin_img = pygame.transform.scale(pin_img, (50, 50))

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.SysFont("arial", 32)
BIG_FONT = pygame.font.SysFont("arial", 48)

BALLOON_IMAGES = [
    pygame.transform.scale(pygame.image.load("red_balloon.png").convert_alpha(), (120, 160)),
    pygame.transform.scale(pygame.image.load("blue_balloon.png").convert_alpha(), (120, 160)),
    pygame.transform.scale(pygame.image.load("golden_balloon.png").convert_alpha(), (120, 160)),
    pygame.transform.scale(pygame.image.load("pink_balloon.png").convert_alpha(), (120, 160)),
    pygame.transform.scale(pygame.image.load("golden_balloon.png").convert_alpha(), (120, 160))
]

# --- Loud Sound ---
pop_sound = pygame.mixer.Sound("pop.wav")

# --- Background image ---
background_img = pygame.image.load("background.jpg").convert()
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))

def draw_text_with_bg(win, text_surface, x, y, padding=10, bg_color=(255, 255, 255, 180)):
    rect = text_surface.get_rect(topleft=(x, y))
    background = pygame.Surface((rect.width + padding*2, rect.height + padding*2), pygame.SRCALPHA)
    background.fill(bg_color)
    win.blit(background, (x - padding, y - padding))
    win.blit(text_surface, (x, y))



# --- Balloon Class ---
class Balloon:
    def __init__(self):
        self.image = random.choice(BALLOON_IMAGES)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.reset()

    def reset(self):
        self.x = random.randint(0, WIDTH - self.width)
        self.y = HEIGHT + random.randint(0, 300)  # Start off-screen below
        self.speed = random.uniform(0.75, 1.75)
        self.popped = False

    def update(self):
        if not self.popped:
            self.y -= self.speed
            if self.y + self.height < 0:  # Balloon floated off-screen
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

    def is_clicked(self, pos):
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        dist = math.hypot(pos[0] - center_x, pos[1] - center_y)
        return dist <= self.width // 2  # Click within radius

# --- Game Variables ---
balloons = [Balloon() for _ in range(3)]
score = 0
GAME_DURATION = 30  # seconds
start_time = pygame.time.get_ticks()  # milliseconds
game_over = False

# --- Main Loop ---
running = True
while running:
    CLOCK.tick(FPS)
    WIN.blit(background_img, (0, 0))

    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # in seconds
    time_left = max(0, GAME_DURATION - int(elapsed_time))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break   

        if not game_over and event.type == pygame.MOUSEBUTTONDOWN:
            for b in balloons:
                if not b.popped and b.is_clicked(event.pos):
                    b.popped = True
                    score += 1
                    pop_sound.play()
        elif game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset game variables
                score = 0
                start_time = pygame.time.get_ticks()
                game_over = False
                for b in balloons:
                    b.reset()
            
            if event.key == pygame.K_ESCAPE:
                running = False

                    
    if not game_over:
  # Update and draw balloons
        for b in balloons:
            if b.popped:
                b.reset()  # Respawn popped balloon
            b.update()
            b.draw(WIN)

   # Draw timer and score
        timer_text = FONT.render(f"Time Left: {time_left}s", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        draw_text_with_bg(WIN, timer_text, 20, 20)
        draw_text_with_bg(WIN, score_text, 20, 60)


        if elapsed_time >= GAME_DURATION:
            game_over = True
            end_time = pygame.time.get_ticks()
    else:
        # Show Game Over screen
        final_text = BIG_FONT.render("Game Over!", True, (200, 0, 0))
        final_score = FONT.render(f"Your Final Score: {score}", True, BLACK)
        restart_hint = FONT.render("Press R to replay or ESC to exit the game", True, BLACK)
        draw_text_with_bg(WIN, final_text, WIDTH//2 - final_text.get_width()//2, HEIGHT//2 - 80)
        draw_text_with_bg(WIN, final_score, WIDTH//2 - final_score.get_width()//2, HEIGHT//2 - 20)
        draw_text_with_bg(WIN, restart_hint, WIDTH//2 - restart_hint.get_width()//2, HEIGHT//2 + 30)

        
    mouse_x, mouse_y = pygame.mouse.get_pos()
    WIN.blit(pin_img, (mouse_x - pin_img.get_width() // 2, mouse_y - pin_img.get_height() // 2))
    pygame.display.update()


pygame.quit()
sys.exit()
