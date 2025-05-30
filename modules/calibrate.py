import cv2
import numpy as np

def compute_homography(camera_points, projector_points):
    cam_pts = np.array(camera_points, dtype=np.float32)
    proj_pts = np.array(projector_points, dtype=np.float32)
    homography, status = cv2.findHomography(cam_pts, proj_pts)
    return homography

def test_homography(homography, point):
    point = np.array([[point]], dtype='float32')
    projected_point = cv2.perspectiveTransform(point, homography)
    return projected_point[0][0]
