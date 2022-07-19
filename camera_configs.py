# filename: camera_configs.py
import cv2
import numpy as np

left_camera_matrix = np.array([[544.73686, 0., 310.61089],
                               [0., 545.21789, 240.69761],
                               [0., 0., 1.]])
left_distortion = np.array([[0.03688, -0.04322, -0.00021, 0.00014, 0.00000]])



right_camera_matrix = np.array([[545.52447, 0., 310.93936],
                                [0., 545.92581, 243.07911],
                                [0., 0., 1.]])
right_distortion = np.array([[0.02876, -0.00089, 0.00033, 0.00116, 0.00000]])

om = np.array([0.00483, -0.00076, -0.00013]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([2.14053, -2.36961, 2.06156]) # 平移关系向量

size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R, T)

# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)