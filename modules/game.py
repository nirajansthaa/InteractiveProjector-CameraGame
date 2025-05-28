import cv2
import numpy as np
import random

# Balloon image files
balloon_paths = ['balloon1.png', 'balloon2.png', 'balloon3.png']
screen_width, screen_height = 800, 600

# Load balloon images and create masks
class Balloon:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # RGBA
        self.image = cv2.resize(self.image, (80, 120))

        # Separate alpha and RGB
        self.bgr = self.image[:, :, :3]
        self.alpha = self.image[:, :, 3]

        # Mask for contour detection
        self.mask = cv2.threshold(self.alpha, 1, 255, cv2.THRESH_BINARY)[1]
        self.contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bounding_box = cv2.boundingRect(self.contours[0])

        # Starting position
        self.x = random.randint(0, screen_width - self.bgr.shape[1])
        self.y = screen_height + random.randint(0, 300)

        # Speed
        self.speed = random.randint(2, 5)

    def move(self):
        self.y -= self.speed
        if self.y + self.bgr.shape[0] < 0:
            self.reset()

    def reset(self):
        self.x = random.randint(0, screen_width - self.bgr.shape[1])
        self.y = screen_height + random.randint(50, 300)
        self.speed = random.randint(2, 5)

    def draw(self, frame):
        self._draw_on_canvas(frame, show_bbox=True)

    def draw_on_debug_canvas(self, canvas):
        self._draw_on_canvas(canvas, show_bbox=True, contour_on_mask=True)

    def _draw_on_canvas(self, canvas, show_bbox=False, contour_on_mask=False):
        h, w = self.bgr.shape[:2]
        y1, y2 = max(self.y, 0), min(self.y + h, screen_height)
        x1, x2 = max(self.x, 0), min(self.x + w, screen_width)

        if y1 < y2 and x1 < x2:
            y_img1, y_img2 = y1 - self.y, y2 - self.y
            x_img1, x_img2 = x1 - self.x, x2 - self.x

            fg = cv2.bitwise_and(self.bgr[y_img1:y_img2, x_img1:x_img2],
                                 self.bgr[y_img1:y_img2, x_img1:x_img2],
                                 mask=self.alpha[y_img1:y_img2, x_img1:x_img2])
            bg = canvas[y1:y2, x1:x2]
            bg = cv2.bitwise_and(bg, bg,
                                 mask=cv2.bitwise_not(self.alpha[y_img1:y_img2, x_img1:x_img2]))
            canvas[y1:y2, x1:x2] = cv2.add(bg, fg)

        if show_bbox:
            bx, by, bw, bh = self.bounding_box
            cv2.rectangle(canvas,
                          (self.x + bx, self.y + by),
                          (self.x + bx + bw, self.y + by + bh),
                          (0, 0, 255), 2)
        if contour_on_mask:
            shifted_contours = [c + [self.x, self.y] for c in self.contours]
            cv2.drawContours(canvas, shifted_contours, -1, (0, 255, 0), 2)


# Initialize balloons
balloons = [Balloon(path) for path in balloon_paths]

# Game loop
def run_game():
    while True:
        frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # white background
        debug_view = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

        for balloon in balloons:
            balloon.move()
            balloon.draw(frame)
            balloon.draw_on_debug_canvas(debug_view)
        
        # Show the frame
        # cv2.imshow("Balloon Game", frame)
        cv2.imshow("Mask & BBox Debug View", debug_view)

        # Exit on 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()
