import cv2
import numpy as np
import random

class Balloon:
    def __init__(self, img_path, screen_width=800, screen_height=600):
        """
        Initialize a balloon with an image and random starting position.
        
        Args:
            img_path (str): Path to the balloon image file (RGBA).
            screen_width (int): Width of the game screen.
            screen_height (int): Height of the game screen.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.alive = True
        self.popped = False
        self.pop_timer = 0
        self.pop_cooldown = 5
        self.image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # RGBA
        if self.image is None:
            raise ValueError(f"Could not load image: {img_path}")
        self.image = cv2.resize(self.image, (180, 220))

        # Separate alpha and RGB
        #Assuming the image was loaded as RGBA
        self.bgr = self.image[:, :, :3] # selects all rows, all columns, and the first three channels.
        self.alpha = self.image[:, :, 3] # extracts the fourth channel, which is the alpha channel 

        # Mask for contour detection
        self.mask = cv2.threshold(self.alpha, 1, 255, cv2.THRESH_BINARY)[1]
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bounding_box = cv2.boundingRect(self.contours[0])

        # Starting position
        self.reset()

    def move(self):
        """Move the balloon upward and reset if it moves off-screen."""
        if self.popped:
            return
        self.y -= self.speed
        if self.y + self.bgr.shape[0] < 0:
            self.reset()

    def pop(self):
        self.popped = True
        self.pop_timer = self.pop_cooldown

    def update(self):
        if self.popped:
            self.pop_timer -= 1
            if self.pop_timer <= 0:
                self.popped = False
                self.reset()

    def reset(self):
        """Reset balloon to a random position below the screen with a new speed."""
        self.x = random.randint(0, self.screen_width - self.bgr.shape[1])
        self.y = self.screen_height + random.randint(50, 300)
        self.speed = random.randint(2, 5)

    def draw(self, frame, show_bbox=False):
        h, w = self.bgr.shape[:2]
        y1, y2 = max(self.y, 0), min(self.y + h, self.screen_height)
        x1, x2 = max(self.x, 0), min(self.x + w, self.screen_width)
        if y1 >= y2 or x1 >= x2:
            return  # Skip drawing if outside frame
        
        # correspond to the visible part of the balloon on the frame
        y_img1 = y1 - self.y
        y_img2 = y2 - self.y
        x_img1 = x1 - self.x
        x_img2 = x2 - self.x

        # Ensure all slices are valid
        if y_img1 >= y_img2 or x_img1 >= x_img2:
            return

        fg = cv2.bitwise_and(
            self.bgr[y_img1:y_img2, x_img1:x_img2],
            self.bgr[y_img1:y_img2, x_img1:x_img2],
            mask=self.alpha[y_img1:y_img2, x_img1:x_img2]
        )

        bg = frame[y1:y2, x1:x2]

        if fg.shape != bg.shape:
            return  # Avoid crash due to shape mismatch

        bg = cv2.bitwise_and(
            bg, bg, mask=cv2.bitwise_not(self.alpha[y_img1:y_img2, x_img1:x_img2])
        )

        frame[y1:y2, x1:x2] = cv2.add(bg, fg)

        if show_bbox:
            bx, by, bw, bh = self.bounding_box
            cv2.rectangle(
                frame,
                (self.x + bx, self.y + by),
                (self.x + bx + bw, self.y + by + bh),
                (0, 0, 255), 2
            )


    def check_hit(self, hit_x, hit_y):
        """
        Check if a point hits the balloon.
        
        Args:
            hit_x (int): X-coordinate of the hit point.
            hit_y (int): Y-coordinate of the hit point.
        
        Returns:
            bool: True if the point hits the balloon, False otherwise.
        """
        bx, by, bw, bh = self.bounding_box
        if self.x + bx <= hit_x <= self.x + bx + bw and self.y + by <= hit_y <= self.y + by + bh:
            mask_y = hit_y - self.y
            mask_x = hit_x - self.x
            if 0 <= mask_x < self.mask.shape[1] and 0 <= mask_y < self.mask.shape[0]:
                if self.mask[mask_y, mask_x] > 0:
                    self.reset()
                    return True
        return False