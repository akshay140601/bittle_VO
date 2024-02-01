import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os


with open("config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

rgb_value = config['parameters']['rgb']
rectified_value = config['parameters']['rectified']
detector_name = config['parameters']['detector']
max_depth_value = config['parameters']['max_depth']

def decomposition(p):

    intrinsic_matrix, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(
        p)

    translation_vector = (translation_vector / translation_vector[3])[:3]

    return intrinsic_matrix, rotation_matrix, translation_vector

def disparity_mapping(left_image, right_image, rgb=rgb_value):
    if rgb:
        num_channels = 3
    else:
        num_channels = 1

    num_disparities = 6*16
    block_size = 7

    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * num_channels * block_size ** 2,
                                    P2=32 * num_channels * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )
    if rgb:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_image_disparity_map = matcher.compute(
        left_image, right_image).astype(np.float32)/16

    return left_image_disparity_map


def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified=rectified_value):

    focal_length = left_intrinsic[0][0]


    if rectified:
        baseline = right_translation[0] - left_translation[0]
    else:
        baseline = left_translation[0] - right_translation[0]

    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    depth_map = np.ones(left_disparity_map.shape)
    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map


def stereo_depth(left_image, right_image, P0, P1, rgb=rgb_value):

    disp_map = disparity_mapping(left_image,
                                 right_image,
                                 rgb=rgb)

    l_intrinsic, _, l_translation = decomposition(
        P0)
    _, _, r_translation = decomposition(
        P1)

    depth = depth_mapping(disp_map, l_intrinsic, l_translation, r_translation)

    return depth

def feature_extractor(image, detector=detector_name, mask=None):
    if detector == 'sift':
        create_detector = cv2.SIFT_create()
    elif detector == 'orb':
        create_detector = cv2.ORB_create()

    keypoints, descriptors = create_detector.detectAndCompute(image, mask)

    return keypoints, descriptors


def feature_matching(first_descriptor, second_descriptor, detector=detector_name, k=2,  distance_threshold=1.0):

    if detector == 'sift':
        feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
        feature_matcher = cv2.BFMatcher_create(
            cv2.NORM_L2, crossCheck=False)
    matches = feature_matcher.knnMatch(
        first_descriptor, second_descriptor, k=k)

    filtered_matches = []
    for match1, match2 in matches:
        if match1.distance <= distance_threshold * match2.distance:
            filtered_matches.append(match1)

    return filtered_matches


def visualize_matches(first_image, second_image, keypoint_one, keypoint_two, matches):

    show_matches = cv2.drawMatches(
        first_image, keypoint_one, second_image, keypoint_two, matches, None, flags=2)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.imshow(show_matches)
    plt.show()

def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, max_depth=max_depth_value):
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    image1_points = np.float32(
        [firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32(
        [secondImage_keypoints[m.trainIdx].pt for m in matches])

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

        if z > max_depth:
            outliers.append(indices)
            continue

        x = z * (u - cx) / fx
        y = z * (v - cy)/ fy

        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

def get_fpath(frame_number):
    dataset_fpath = 'dataset/sequences/00'
    frame_num = '{:06d}'.format(frame_number)
    left_fpath = os.path.join(dataset_fpath, 'image_0',
                              f'{frame_num}.png')
    
    return left_fpath

def essential_matrix(kp1, kp2, method=cv2.RANSAC):
    E, _ = cv2.findEssentialMat(kp1, kp2, method=method)
    return E

def recover_pose(E, kp1, kp2):
    _, R, t, _ = cv2.recoverPose(E, kp1, kp2)
    return R, t