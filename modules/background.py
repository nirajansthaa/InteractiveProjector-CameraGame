import pygame

def draw_text_with_bg(win, text_surface, x, y, padding=10, bg_color=(255, 255, 255, 180)):
    rect = text_surface.get_rect(topleft=(x, y))
    background = pygame.Surface((rect.width + padding*2, rect.height + padding*2), pygame.SRCALPHA)
    background.fill(bg_color)
    win.blit(background, (x - padding, y - padding))
    win.blit(text_surface, (x, y))