import cv2
import numpy as np
import os
import time
from screeninfo import get_monitors
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.camera_capture import capture_camera_points
from modules.projector_display import show_calibration_pattern
from modules.calibrate import compute_homography
from modules.game_visual import Balloon
from modules.circle_detection import detect_circles_in_frame
from modules.hit_detection import HitDetector

# Configuration
CAMERA_ID = 1
SCREEN_WIDTH, SCREEN_HEIGHT = 1900, 1000  # Projector resolution
TARGET_FPS = 30

# Circle detection parameters
DETECTION_PARAMS = {
    'min_radius': 30,
    'max_radius': 50,
    'min_dist': 80,
    'param1': 120,
    'param2': 30,
    'intensity_threshold': 100
}

# Optimization settings
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
USE_BACKGROUND_SUBTRACTION = True
HOMOGRAPHY_ERROR_THRESHOLD = 15.0  # Maximum acceptable reprojection error


class FrameProcessor:
    """Optimized frame processor with motion detection and frame skipping"""
    
    def __init__(self, process_every_n_frames=2, use_background_subtraction=True):
        self.process_every_n_frames = process_every_n_frames
        self.use_background_subtraction = use_background_subtraction
        self.frame_count = 0
        self.last_circles = None
        
        # Background subtractor for motion detection
        if use_background_subtraction:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=50,
                history=500
            )
            self.motion_threshold = 1000  # Minimum motion pixels to trigger processing
        
    def should_process_frame(self, frame):
        """Determine if current frame should be processed"""
        self.frame_count += 1
        
        # Skip frames based on frequency
        if self.frame_count % self.process_every_n_frames != 0:
            return False
        
        # If using background subtraction, check for motion
        if self.use_background_subtraction:
            return self._has_significant_motion(frame)
        
        return True
    
    def _has_significant_motion(self, frame):
        """Check if frame has significant motion using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > self.motion_threshold
    
    def get_cached_circles(self):
        """Return last detected circles when skipping frame processing"""
        return self.last_circles
    
    def update_circles(self, circles):
        """Update the cached circles from latest processing"""
        self.last_circles = circles


def validate_homography_quality(camera_points, projector_points, homography, threshold=5.0):
    """Validate homography quality using reprojection error"""
    try:
        # Convert points to proper format for perspective transform
        camera_pts = np.array(camera_points, dtype='float32').reshape(-1, 1, 2)
        
        # Project camera points to projector space using homography
        projected_points = cv2.perspectiveTransform(camera_pts, homography)
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate reprojection errors
        projector_pts = np.array(projector_points, dtype='float32')
        errors = np.sqrt(np.sum((projected_points - projector_pts) ** 2, axis=1))
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # Validation criteria
        is_valid = mean_error < threshold and max_error < (threshold * 2)
        
        print(f"Homography Validation:")
        print(f"  Mean reprojection error: {mean_error:.2f} pixels")
        print(f"  Max reprojection error: {max_error:.2f} pixels")
        print(f"  Quality: {'GOOD' if is_valid else 'POOR'}")
        
        return is_valid, mean_error, errors
    except Exception as e:
        print(f"Error during homography validation: {e}")
        return False, float('inf'), []


def check_homography_properties(homography):
    """Additional checks for homography matrix properties"""
    try:
        # Check if matrix is invertible
        det = np.linalg.det(homography)
        if abs(det) < 1e-6:
            print("Warning: Homography matrix is nearly singular")
            return False
        
        # Check condition number (stability)
        cond_num = np.linalg.cond(homography)
        if cond_num > 1000:
            print(f"Warning: Homography is poorly conditioned (condition number: {cond_num:.2f})")
            return False
        
        # Check for reasonable transformation (not too extreme scaling/rotation)
        h = homography
        scale_x = np.sqrt(h[0,0]**2 + h[1,0]**2)
        scale_y = np.sqrt(h[0,1]**2 + h[1,1]**2)
        
        if scale_x < 0.1 or scale_x > 10 or scale_y < 0.1 or scale_y > 10:
            print(f"Warning: Extreme scaling detected (scale_x: {scale_x:.2f}, scale_y: {scale_y:.2f})")
            return False
        
        print("Homography matrix properties are acceptable")
        return True
    except Exception as e:
        print(f"Error checking homography properties: {e}")
        return False


def compute_homography_with_validation(camera_points, projector_points, max_attempts=3):
    """Compute homography with quality validation and multiple attempts"""
    for attempt in range(max_attempts):
        try:
            # Use RANSAC for better robustness
            homography_result = cv2.findHomography(
                np.array(camera_points, dtype='float32'),
                np.array(projector_points, dtype='float32'),
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if homography_result is None or homography_result[0] is None:
                print(f"Attempt {attempt + 1}: Homography computation failed")
                continue
            
            homography = homography_result[0]
            
            # Check matrix properties
            if not check_homography_properties(homography):
                print(f"Attempt {attempt + 1}: Poor homography matrix properties")
                continue
            
            # Validate quality
            is_valid, error, _ = validate_homography_quality(
                camera_points, projector_points, homography, HOMOGRAPHY_ERROR_THRESHOLD
            )
            
            if is_valid:
                print(f"Homography computed successfully on attempt {attempt + 1}")
                return homography
            else:
                print(f"Attempt {attempt + 1}: Poor homography quality (error: {error:.2f})")
                
        except cv2.error as e:
            print(f"Attempt {attempt + 1}: Error computing homography: {e}")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Unexpected error: {e}")
    
    print("Failed to compute good quality homography after all attempts")
    return None


def detect_circles_optimized(frame, frame_processor, detection_params):
    """Optimized circle detection that may skip processing based on frame processor"""
    should_process = frame_processor.should_process_frame(frame)
    
    if should_process:
        # Process current frame
        circles, processed_frame = detect_circles_in_frame(frame, **detection_params)
        frame_processor.update_circles(circles)
        return circles, processed_frame, True
    else:
        # Use cached results
        circles = frame_processor.get_cached_circles()
        return circles, frame, False


def initialize_system():
    """Initialize camera and monitor setup"""
    # Get monitor info for projector and camera feed windows
    monitors = get_monitors()
    if not monitors:
        print("Error: No monitors detected")
        return None, None, None
    
    main_screen = monitors[0]
    external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
    main_origin = (main_screen.x, main_screen.y)
    external_origin = (external_screen.x, external_screen.y)
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {CAMERA_ID}")
        return None, None, None
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    return cap, main_origin, external_origin


def load_balloon_assets():
    """Load balloon images from assets directory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    balloon_paths = [
        os.path.join(base_dir, 'assets', 'balloon1.png'),
        os.path.join(base_dir, 'assets', 'balloon2.png'),
        os.path.join(base_dir, 'assets', 'balloon3.png')
    ]
    
    balloons = []
    for path in balloon_paths:
        if not os.path.exists(path):
            print(f"Warning: Image file {path} does not exist. Skipping.")
            continue
        try:
            balloons.append(Balloon(path, SCREEN_WIDTH, SCREEN_HEIGHT))
            print(f"Loaded balloon image: {path}")
        except ValueError as e:
            print(f"Warning: {e}. Skipping image {path}.")
    
    if not balloons:
        print("Error: No valid balloon images loaded.")
        return None
    
    return balloons
import cv2
import numpy as np
import math

def analyze_quadrilateral_quality(points):
    """
    Analyze the quality of calibration points as a quadrilateral
    Returns quality score and diagnostic information
    """
    if len(points) != 4:
        return 0, "Need exactly 4 points"
    
    points = np.array(points, dtype='float32')
    
    # Calculate side lengths
    sides = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        side_length = np.linalg.norm(p2 - p1)
        sides.append(side_length)
    
    # Calculate area using shoelace formula
    area = 0.5 * abs(sum(points[i][0] * points[(i+1)%4][1] - points[(i+1)%4][0] * points[i][1] for i in range(4)))
    
    # Calculate angles at each corner
    angles = []
    for i in range(4):
        p1 = points[(i-1) % 4]
        p2 = points[i]
        p3 = points[(i+1) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        angle = math.degrees(math.acos(cos_angle))
        angles.append(angle)
    
    # Quality checks
    diagnostics = []
    quality_score = 100
    
    # Check if quadrilateral is too small
    min_area = 50000  # Minimum area threshold
    if area < min_area:
        quality_score -= 30
        diagnostics.append(f"Quadrilateral too small (area: {area:.0f}, min: {min_area})")
    
    # Check side length ratios (should be reasonably balanced)
    min_side, max_side = min(sides), max(sides)
    side_ratio = max_side / min_side if min_side > 0 else float('inf')
    if side_ratio > 3:
        quality_score -= 25
        diagnostics.append(f"Unbalanced sides (ratio: {side_ratio:.1f})")
    
    # Check for very acute or very obtuse angles
    for i, angle in enumerate(angles):
        if angle < 30 or angle > 150:
            quality_score -= 20
            diagnostics.append(f"Poor angle at corner {i+1}: {angle:.1f}°")
    
    # Check if points form a convex quadrilateral
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Check convexity by ensuring all cross products have same sign
    cross_products = []
    for i in range(4):
        cp = cross_product(points[i], points[(i+1)%4], points[(i+2)%4])
        cross_products.append(cp)
    
    if not (all(cp > 0 for cp in cross_products) or all(cp < 0 for cp in cross_products)):
        quality_score -= 40
        diagnostics.append("Points don't form a convex quadrilateral")
    
    # Check for self-intersection
    def lines_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    # Check opposite sides for intersection
    if lines_intersect(points[0], points[1], points[2], points[3]):
        quality_score -= 50
        diagnostics.append("Quadrilateral is self-intersecting")
    
    return max(0, quality_score), diagnostics


def provide_calibration_guidance(points, target_points):
    """Provide specific guidance for improving calibration points"""
    if len(points) < 4:
        return f"Click {4 - len(points)} more point(s). Try to form a large rectangle."
    
    score, diagnostics = analyze_quadrilateral_quality(points)
    
    guidance = []
    guidance.append(f"Calibration Quality: {score}/100")
    
    if score < 70:
        guidance.append("\nImprovements needed:")
        for diag in diagnostics:
            guidance.append(f"  • {diag}")
        
        guidance.append("\nTips for better calibration:")
        guidance.append("  • Click points to form a large, well-spaced rectangle")
        guidance.append("  • Avoid very small or very skewed quadrilaterals")
        guidance.append("  • Make sure corners are roughly 90 degrees")
        guidance.append("  • Use most of the camera's field of view")
        guidance.append("  • Points should be in clockwise or counter-clockwise order")
    else:
        guidance.append("\nGood calibration points!")
    
    return "\n".join(guidance)


def enhanced_capture_camera_points(resolution):
    """Enhanced camera point capture with real-time feedback"""
    global clicked_points
    clicked_points = []
    
    # Capture from camera
    cap = cv2.VideoCapture(1)  # Adjust camera ID as needed
    if not cap.isOpened():
        print("Error: Could not open camera")
        return []
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Clicked: {x}, {y}")
    
    cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration", resolution[0], resolution[1])
    cv2.setMouseCallback("Camera Calibration", mouse_callback)
    
    print("\nCalibration Instructions:")
    print("1. Click on 4 corners of the projection area in the camera feed")
    print("2. Try to form a large, well-spaced rectangle")
    print("3. Click corners in order (clockwise or counter-clockwise)")
    print("4. Press SPACE when done, 'r' to reset, ESC to cancel")
    
    target_points = [(100, 100), (1800, 100), (1800, 900), (100, 900)]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, resolution)
        frame = cv2.flip(frame, 1)
        
        # Draw clicked points
        for i, point in enumerate(clicked_points):
            cv2.circle(frame, point, 10, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (point[0]+15, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw lines between points if we have multiple points
        if len(clicked_points) > 1:
            for i in range(len(clicked_points)):
                start = clicked_points[i]
                end = clicked_points[(i + 1) % len(clicked_points)] if len(clicked_points) > 2 else clicked_points[-1]
                if i < len(clicked_points) - 1 or len(clicked_points) == 4:
                    cv2.line(frame, start, end, (255, 0, 0), 2)
        
        # Show guidance text
        guidance = provide_calibration_guidance(clicked_points, target_points)
        
        # Display guidance on frame
        y_offset = 30
        for line in guidance.split('\n')[:5]:  # Show first 5 lines on frame
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        # Print full guidance to console when points change
        if len(clicked_points) == 4:
            print(f"\n{guidance}")
        
        cv2.imshow("Camera Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            clicked_points = []
            break
        elif key == ord('r'):  # Reset
            clicked_points = []
            print("Points reset. Click 4 new points.")
        elif key == ord(' ') and len(clicked_points) == 4:  # Space
            score, _ = analyze_quadrilateral_quality(clicked_points)
            if score >= 50:  # Lower threshold but still reasonable
                print("Calibration points accepted!")
                break
            else:
                print(f"Points quality too low ({score}/100). Please improve or press 'r' to reset.")
    
    cap.release()
    cv2.destroyWindow("Camera Calibration")
    return clicked_points


def robust_homography_computation(camera_points, projector_points):
    """More robust homography computation with multiple methods"""
    
    # First, let's analyze and potentially reorder points
    camera_points, projector_points = reorder_points_if_needed(camera_points, projector_points)
    
    methods = [
        (cv2.RANSAC, "RANSAC"),
        (cv2.LMEDS, "LMEDS"),
        (0, "Standard")  # 0 means standard method
    ]
    
    best_homography = None
    best_score = float('inf')
    
    for method, method_name in methods:
        try:
            if method == 0:
                homography = cv2.findHomography(
                    np.array(camera_points, dtype='float32'),
                    np.array(projector_points, dtype='float32')
                )[0]
            else:
                homography = cv2.findHomography(
                    np.array(camera_points, dtype='float32'),
                    np.array(projector_points, dtype='float32'),
                    method=method,
                    ransacReprojThreshold=5.0
                )[0]
            
            if homography is not None:
                # Evaluate this homography
                is_valid, error, _ = validate_homography_quality(
                    camera_points, projector_points, homography, threshold=10.0  # More lenient
                )
                
                condition_number = np.linalg.cond(homography)
                
                print(f"{method_name} method: error={error:.2f}, condition={condition_number:.1f}")
                
                # Score based on error and condition number
                score = error + (condition_number / 1000)  # Weight condition number less
                
                if score < best_score and condition_number < 50000:  # More lenient condition
                    best_score = score
                    best_homography = homography
                    print(f"  -> New best homography (score: {score:.2f})")
        
        except Exception as e:
            print(f"{method_name} method failed: {e}")
    
    if best_homography is not None:
        print(f"Selected homography with score: {best_score:.2f}")
    
    return best_homography


def reorder_points_if_needed(camera_points, projector_points):
    """Reorder points to ensure consistent mapping"""
    # Convert to numpy arrays
    cam_pts = np.array(camera_points, dtype='float32')
    proj_pts = np.array(projector_points, dtype='float32')
    
    # Find the centroid of camera points
    centroid = np.mean(cam_pts, axis=0)
    
    # Calculate angles from centroid to each point
    angles = []
    for point in cam_pts:
        angle = math.atan2(point[1] - centroid[1], point[0] - centroid[0])
        angles.append(angle)
    
    # Sort points by angle (counter-clockwise from right)
    sorted_indices = sorted(range(4), key=lambda i: angles[i])
    
    # Reorder both camera and projector points
    reordered_camera = [camera_points[i] for i in sorted_indices]
    
    # For projector points, use standard order: top-left, top-right, bottom-right, bottom-left
    standard_projector = [(100, 100), (1800, 100), (1800, 900), (100, 900)]
    
    return reordered_camera, standard_projector


def enhanced_calibrate_system():
    """Enhanced calibration system with better validation"""
    print("Starting enhanced system calibration...")
    
    # Step 1: Show calibration pattern on projector
    projector_points = show_calibration_pattern(screen_resolution=(1900, 1000))
    
    # Step 2: Enhanced camera point capture with guidance
    camera_points = enhanced_capture_camera_points(resolution=(1900, 1000))
    
    if len(camera_points) != 4:
        print("Calibration cancelled or failed.")
        return None
    
    # Step 3: Analyze point quality one more time
    score, diagnostics = analyze_quadrilateral_quality(camera_points)
    print(f"\nFinal calibration quality: {score}/100")
    if diagnostics:
        for diag in diagnostics:
            print(f"  • {diag}")
    
    # Step 4: Robust homography computation
    print("\nComputing homography with multiple methods...")
    homography = robust_homography_computation(camera_points, projector_points)
    
    if homography is not None:
        print("Calibration completed successfully!")
        
        # Final validation
        is_valid, error, _ = validate_homography_quality(
            camera_points, projector_points, homography, threshold=15.0
        )
        print(f"Final reprojection error: {error:.2f} pixels")
        
    else:
        print("Calibration failed after trying all methods.")
        print("\nTroubleshooting tips:")
        print("  • Ensure the 4 points form a large, clear rectangle")
        print("  • Make sure points are in the projected area")
        print("  • Avoid clicking very close to edges or corners")
        print("  • Try to use most of the camera's field of view")
        print("  • Ensure good lighting and contrast")
    
    return homography

def calibrate_system():
    return enhanced_calibrate_system()    


def setup_windows(main_origin, external_origin):
    """Create and position display windows"""
    cv2.namedWindow("Game Display", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Game Display", external_origin[0], external_origin[1])
    cv2.moveWindow("Camera Feed", main_origin[0], main_origin[1])
    cv2.resizeWindow("Game Display", SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.resizeWindow("Camera Feed", SCREEN_WIDTH, SCREEN_HEIGHT)


def hit_callback(x, y, r, balloon):
    """Callback function when balloon is hit"""
    balloon.pop()
    print(f"Balloon popped at ({x}, {y})")


def cleanup_resources(cap):
    """Clean up resources"""
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function"""
    print("Initializing Interactive Balloon Game...")
    
    # Initialize system components
    cap, main_origin, external_origin = initialize_system()
    if cap is None:
        return
    
    # Load balloon assets
    balloons = load_balloon_assets()
    if balloons is None:
        cleanup_resources(cap)
        return
    
    # Initialize hit detector
    hit_detector = HitDetector()
    
    # Initialize frame processor for optimization
    frame_processor = FrameProcessor(
        process_every_n_frames=PROCESS_EVERY_N_FRAMES,
        use_background_subtraction=USE_BACKGROUND_SUBTRACTION
    )
    
    # Setup display windows
    setup_windows(main_origin, external_origin)
    
    try:
        # Run calibration
        homography = calibrate_system()
        if homography is None:
            print("Calibration failed. Exiting.")
            return
        
        print("Starting game loop... Press 'q' or ESC to quit.")
        
        # Game loop
        frame_time = 1.0 / TARGET_FPS
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            start_time = time.time()
            
            # Create white background for game display
            game_frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255
            
            # Read camera frame
            ret, camera_frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Prepare camera frame
            camera_frame = cv2.resize(camera_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            camera_frame = cv2.flip(camera_frame, 1)  # Flip for consistency
            
            # Update and draw balloons
            for balloon in balloons:
                balloon.update()
                balloon.move()
                balloon.draw(game_frame, show_bbox=False)
            
            # Optimized circle detection
            circles, processed_frame, was_processed = detect_circles_optimized(
                camera_frame, frame_processor, DETECTION_PARAMS
            )
            
            # Debug visualization
            processing_status = "PROCESSED" if was_processed else "CACHED"
            cv2.putText(processed_frame, f"Status: {processing_status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                fps_elapsed = time.time() - fps_start_time
                fps = 30 / fps_elapsed if fps_elapsed > 0 else 0
                fps_start_time = time.time()
                
            if 'fps' in locals():
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Transform and visualize circles
            debug_visualization = True
            if circles is not None and isinstance(circles, np.ndarray) and circles.size > 0:
                circles = np.round(circles[0, :]).astype("int")
                if circles.ndim == 1:
                    circles = circles.reshape(1, -1)
                
                # Transform circles to projector coordinates
                circles_transformed = []
                for (x, y, r) in circles:
                    try:
                        point = np.array([[x, y]], dtype='float32')
                        projected_point = cv2.perspectiveTransform(point[None, :, :], homography)[0][0]
                        x_p, y_p = int(projected_point[0]), int(projected_point[1])
                        
                        # Check if projected point is within bounds
                        if 0 <= x_p < SCREEN_WIDTH and 0 <= y_p < SCREEN_HEIGHT:
                            circles_transformed.append([x_p, y_p, r])
                            
                            if debug_visualization:
                                cv2.circle(game_frame, (x_p, y_p), r, (0, 255, 255), 3)  # Yellow outline
                                cv2.circle(game_frame, (x_p, y_p), 2, (0, 0, 255), 3)    # Red center
                    except Exception as e:
                        print(f"Error transforming circle: {e}")
                        continue
                
                if circles_transformed:
                    circles_transformed = np.array(circles_transformed, dtype="int")
                    hit_detector.process_detected_circles(circles_transformed, balloons, callback=hit_callback)
            
            # Display frames
            processed_frame = cv2.resize(processed_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            cv2.imshow("Game Display", game_frame)
            cv2.imshow("Camera Feed", processed_frame)
            
            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(1, int((frame_time - elapsed) * 1000))
            key = cv2.waitKey(sleep_time) & 0xFF
            
            if key in [ord('q'), 27]:  # 'q' or ESC key
                print("Exiting game...")
                break
            elif key == ord('r'):  # 'r' key to recalibrate
                print("Recalibrating system...")
                new_homography = calibrate_system()
                if new_homography is not None:
                    homography = new_homography
                    print("Recalibration successful!")
                else:
                    print("Recalibration failed, using previous calibration.")
    
    except KeyboardInterrupt:
        print("Game interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup_resources(cap)
        print("Game ended. Resources cleaned up.")


if __name__ == "__main__":
    main()