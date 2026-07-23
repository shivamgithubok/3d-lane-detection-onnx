import numpy as np

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi/2 + cam_pitch), -np.sin(np.pi/2 + cam_pitch), cam_height],
        [0, np.sin(np.pi/2 + cam_pitch),  np.cos(np.pi/2 + cam_pitch), 0]
    ])
    return K @ P_g2c

def homography_crop_resize(org_hw, crop_y, resize_hw):
    ratio_x = resize_hw[1] / org_hw[1]
    ratio_y = resize_hw[0] / (org_hw[0] - crop_y)
    return np.array([[ratio_x, 0, 0],
                     [0, ratio_y, -ratio_y * crop_y],
                     [0, 0, 1]])

# same generic assumptions Anchor3DLane's own config already uses,
# tuned for a forward-facing dashcam roughly this height/tilt
CAM_HEIGHT = 1.55       # meters
CAM_PITCH_DEG = 3       # degrees, slight downward tilt
K = np.array([[2015.0, 0.0, 960.0],
              [0.0, 2015.0, 540.0],
              [0.0, 0.0, 1.0]])   # generic intrinsics matched to a 1920x1280-ish sensor

P_g2im = projection_g2im(np.radians(CAM_PITCH_DEG), CAM_HEIGHT, K)
H_crop = homography_crop_resize([1280, 1920], 0, [360, 480])
P_final = H_crop @ P_g2im
print(P_final.tolist())