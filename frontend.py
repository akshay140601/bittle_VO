import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import stereo_depth, decomposition, feature_extractor, feature_matching, motion_estimation, get_fpath, essential_matrix, recover_pose
#from backend import levenberg_marquardt_optimization, Powells_dog_leg_optimization, ISAM2, LM
#from bow import BoW
import cv2

# path for the bow vocabulary file
vocab_path = 'vocab.npy';

with open("config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

detector_name = config['parameters']['detector']
subset = config['parameters']['subset']
threshold = config['parameters']['distance_threshold']

bow_offset = config['parameters']['loop_closure']['offset']
bow_stride = config['parameters']['loop_closure']['stride']
bow_threshold = config['parameters']['loop_closure']['threshold']

def frontend(data_handler, detector=detector_name, mask=None, subset=subset, plot=True):

    # init bow
    #bow = BoW();
    #bow.load_vocab(vocab_path);

    if subset is not None:
        num_frames = subset
    else:
        num_frames = data_handler.frames

    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = data_handler.ground_truth[:, 0, 3]
        ys = data_handler.ground_truth[:, 1, 3]
        zs = data_handler.ground_truth[:, 2, 3]

        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='dimgray')
        line_trajectory, = ax.plot([], [], [], c='darkorange')
        line_updated_poses, = ax.plot([], [], [], c='pink')

        plt.pause(1e-32)

    homo_matrix = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]
    updated_trajectory = np.zeros((num_frames, 3, 4))
    updated_trajectory[0] = homo_matrix[:3, :]
    left_instrinsic_matrix, _, _ = decomposition(data_handler.P0)

    data_handler.reset_frames()
    next_image = next(data_handler.left_images)

    loop_closure_frames = []
    loop_closure_frames_final = []
    loop_closure_detected = False
    updated_poses = np.zeros((0, 3, 4))

    for i in range(num_frames - 1):

        image_left = next_image
        #image_right = next(data_handler.right_images)
        next_image = next(data_handler.left_images)

        '''depth = stereo_depth(image_left,
                             image_right,
                             P0=data_handler.P0,
                             P1=data_handler.P1)'''

        kp1, desc1 = feature_extractor(
            image_left, detector, mask)
        kp2, desc2 = feature_extractor(
            next_image, detector, mask)

        matches = feature_matching(desc1, desc2,
                                   detector=detector, distance_threshold=threshold)
        
        '''rotation_matrix, translation_vector, _, _ = motion_estimation(
            matches, keypoint_left_first, keypoint_left_next, left_instrinsic_matrix, depth)'''
        
        # Pose recovery
        ess_mat = essential_matrix(kp1, kp2)
        rotation_matrix, translation_vector = recover_pose(ess_mat, kp1, kp2)


        Transformation_matrix = np.eye(4)

        Transformation_matrix[:3, :3] = rotation_matrix
        Transformation_matrix[:3, 3] = translation_vector.T

        homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
        #print(homo_matrix)

        trajectory[i+1, :, :] = homo_matrix[:3, :]
        updated_trajectory[i+1, :, :] = homo_matrix[:3, :]

        '''left_fpath = get_fpath(i)
        bow.add_frame(left_fpath)
        loop_closure_detected_first = bow.is_loop_closure(bow_offset, bow_stride, bow_threshold, loop_closure_frames)
        #print(loop_closure_frames)
        if loop_closure_detected_first == True:
            print('Passed the first check')
            img1 = cv2.imread(get_fpath(loop_closure_frames[-2]))
            img2 = cv2.imread(get_fpath(loop_closure_frames[-1]))
            kp1, desc1 = feature_extractor(img1, detector, mask=None)
            kp2, desc2 = feature_extractor(img2, detector, mask=None)
            matches = feature_matching(desc1,
                                desc2,
                                detector=detector,
                                distance_threshold=threshold)
            
            if len(matches) > 200:
                print('Passed the second check')
                loop_closure_detected = True
                print('Loop closure detected... Triggering backend optimization...')
                loop_closure_frames_final.append(loop_closure_frames[-2])
                loop_closure_frames_final.append(loop_closure_frames[-1])


        if loop_closure_detected == True:
            print(loop_closure_frames_final)
            updated_poses = LM(updated_trajectory[:i+2, :, :], loop_closure_frames_final)
            updated_trajectory[:i+2, :, :] = updated_poses
            homo_matrix = updated_trajectory[i+1, :, :]
            loop_closure_detected = False

            xs_updated = updated_trajectory[:i+2, 0, 3]
            ys_updated = updated_trajectory[:i+2, 1, 3]
            zs_updated = updated_trajectory[:i+2, 2, 3]
            if line_updated_poses:
                line_updated_poses.remove()
            line_updated_poses, = ax.plot(xs_updated, ys_updated, zs_updated, c='blue')'''

        if i % 10 == 0:
            print(f'{i} frames have been computed')

        if plot:
            xs = trajectory[:i + 2, 0, 3]
            ys = trajectory[:i + 2, 1, 3]
            zs = trajectory[:i + 2, 2, 3]
            line_trajectory.set_data(xs, ys)
            line_trajectory.set_3d_properties(zs)

            plt.pause(1e-32)

    if plot:
        plt.show()

    return trajectory
