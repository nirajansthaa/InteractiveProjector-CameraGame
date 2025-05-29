import math

class HitDetector:
    """
    A class to handle collision detection between circles and balloons.
    """
    def __init__(self):
        pass

    def check_circle_balloon_collision(self, circle_x, circle_y, circle_r, balloon):
        """
        Check if a circle collides with a balloon.
        
        Args:
            circle_x (int): X-coordinate of the circle's center.
            circle_y (int): Y-coordinate of the circle's center.
            circle_r (int): Radius of the circle.
            balloon (Balloon): Balloon object with position and bounding box.
        
        Returns:
            bool: True if the circle collides with the balloon, False otherwise.
        """
        # Get balloon's bounding box (assuming Balloon class has x, y, w, h attributes)
        balloon_x, balloon_y = balloon.x, balloon.y
        # Assume balloon has an 'image' attribute (OpenCV image)
        # Get width and height from the image shape
        try:
            balloon_h, balloon_w = balloon.image.shape[:2]  # shape[0] = height, shape[1] = width
        except AttributeError:
            # Fallback: assume default size if no image attribute
            balloon_w, balloon_h = 50, 50  # Adjust these values based on your balloon sizes
            print("Warning: Balloon has no 'image' attribute; using default size (50x50).")

        # Find the closest point on the balloon's bounding box to the circle's center
        closest_x = max(balloon_x, min(circle_x, balloon_x + balloon_w))
        closest_y = max(balloon_y, min(circle_y, balloon_y + balloon_h))

        # Calculate the distance between the circle's center and the closest point
        distance_x = circle_x - closest_x
        distance_y = circle_y - closest_y
        distance = math.sqrt(distance_x**2 + distance_y**2)

        # Collision occurs if the distance is less than or equal to the circle's radius
        return distance <= circle_r

    def process_detected_circles(self, circles, balloons, callback=None):
        """
        Process a list of detected circles and check for collisions with balloons.
        
        Args:
            circles (list): List of tuples (x, y, r) representing detected circles.
            balloons (list): List of Balloon objects.
            callback (function, optional): Function to call when a collision is detected.
        
        Returns:
            list: List of balloons that were hit.
        """
        hit_balloons = []
        
        for (x, y, r) in circles:
            for balloon in balloons:
                if self.check_circle_balloon_collision(x, y, r, balloon):
                    hit_balloons.append(balloon)
                    if callback:
                        callback(x, y, r, balloon)
        
        return hit_balloons