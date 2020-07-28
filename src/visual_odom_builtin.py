import cv2
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import det
from scipy.ndimage import map_coordinates as interp2
from scipy.optimize import least_squares as ls
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import random
import glob
import math
from math import floor
from operator import itemgetter

import ReadCameraModel
import UndistortImage



def sift_feat(und1, und2):
    '''
    gives out the feature matches
    :param und1: undistorted image 1
    :param und2: undistorted image 2
    :return: matches : obtained matches
    '''

    img1 = und1
    img2 = und2
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    pt=[]
    for m, n in matches:
        if m.distance < 0.2 * n.distance:
            good.append(m)
            pt.append([m,n])
#         if abs(m.distance-n.distance) < 0.01 * m.distance:
#             good.append(m)
#             pt.append([m,n])

    #####################
    img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatchesKnn(img1, kp1, img2, kp2, pt,outImg=img_match, matchColor=None, singlePointColor=(255, 255, 255), flags=2)

    dim = (int(img1.shape[0]/2), int(img1.shape[1]/2))
    img_match=cv2.resize(img_match,dim,interpolation = cv2.INTER_AREA)
#     cv2.imshow("flann matching", img_match)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    #######################
    src_pts = np.asarray([kp1[m[0].queryIdx].pt for m in pt])
    dst_pts = np.asarray([kp2[m[0].trainIdx].pt for m in pt])

    # print(src_pts[0])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    list_1 = src_pts[mask == 1]
    list_2 = dst_pts[mask == 1]
    list_1=np.hstack((list_1,np.ones((np.shape(list_1)[0],1))))
    list_2=np.hstack((list_2,np.ones((np.shape(list_1)[0],1))))

    all_matches = [list_1, list_2]

    return all_matches

def estimate_fundamental_matrix(x_i, x_i_dash):
    '''
    generate the fundamental matrix from correspondences
    :param x_i, x_i_dash: correspondences from image1 and image2
    :return: f_matrix : estimated fundamental matrix
    '''
    A=[]
    for j in range(len(x_i)):
        r =np.array([x_i[j][0]*x_i_dash[j][0], x_i[j][0]*x_i_dash[j][1], x_i[j][0], x_i[j][1]*x_i_dash[j][0], x_i[j][1]*x_i_dash[j][1], x_i[j][1], x_i_dash[j][0], x_i_dash[j][1],1])
        A.append(r)
    A =np.asarray(A)

    ## Obtaining F Estimate
    u, s, vh = np.linalg.svd(A)
    F_est = vh[:,vh.shape[0]-1]
    F_est=np.reshape(F_est,(3,3)).T

    ## Correcting the F_estimate
    u_,s_,vh_ = np.linalg.svd(F_est)
    s_[s_.shape[0]-1] = 0
    s_corrected = np.diag(s_)

    ## Corrected F
    F_tilde = np.matmul(u_, np.matmul(s_corrected, vh_))

    # F_tilde = F_tilde/F_tilde[2,2] ## remove this


    return F_tilde


def RANSAC(all_matches):
    '''
    take all the matches and give out the best matches after rejecting outliers
    :param matches: all the matches obtained from sift feature mapping
    :return: inliers : all the inliers after outlier rejection
    '''
    n=8
    m=10000
    inliers=[]
    Epsilon=0.3
    b = 0
    outliers=[]
    s=[]
    # F_tilde

    for i in range(0,m-1):
        rand_points=random.sample(range(0,len(all_matches[0])),8)
        x_i = list(itemgetter(*rand_points)(all_matches[0]))
        x_i_dash = list(itemgetter(*rand_points)(all_matches[1]))
        F_tilde = estimate_fundamental_matrix(x_i, x_i_dash)


        # x_i = np.asarray(list(itemgetter(*rand_points)(all_matches[0])))
        # x_i_dash = np.asarray(list(itemgetter(*rand_points)(all_matches[1])))
        # F_tilde = NormalizedFMatrix(x_i, x_i_dash)


        for j in range(0,n):
            if abs(np.matmul(np.transpose(x_i_dash[j]),np.matmul(F_tilde,x_i[j])))<Epsilon:
                # print(abs(np.matmul(np.transpose(x_i_dash[j]),np.matmul(F_tilde,x_i[j]))))
                s.append([x_i[j],x_i_dash[j]])
                # print(s)
            else:
                outliers.append([x_i[j],x_i_dash[j]])
#         if len(s)/len(all_matches) > 0.8:
#             inliers.append(s)
        if b<len(s):
            b = len(s)
            inliers = s

    print(F_tilde)

    return inliers,s

    # return inliers
    # pass



def obtain_essential_matrix(f_matrix, calibration_matrix):
    '''
    generate the essential matrix from fundamental matrix
    :param f_matrix: fundamenta lmatrix
    :param calibration_matrix: intrinsic camera calbration matrix
    :return: essential_matrix : essential_matrix
    '''

    essential_matrix = np.matmul(np.transpose(calibration_matrix),np.matmul(f_matrix,calibration_matrix))

    # essential_matrix = essential_matrix/np.linalg.norm(essential_matrix)

    return essential_matrix




def linear_LS_triangulation(u1, P1, u2, P2):

    linear_LS_triangulation_C = -np.eye(2, 3)

    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.zeros((3, len(u1)))

    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)

    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]

        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2

        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1

        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)

    return x.T, np.ones(len(u1), dtype=bool)



def vizMatches(image1, image2, pixelsImg1, pixelsImg2, idx):
    '''
    Visualize the feature match between pair of images
    :param image1:
    :param image2:
    :param pixelsImg1:
    :param pixelsImg2:
    :return:
    '''
    # visualization of the matches
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, :] = image1
    view[:h2, w1:, :] = image2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for ind in range(len(pixelsImg1)):
        # draw the keypoints
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        cv2.line(view, (int(pixelsImg1[ind][0]), int(pixelsImg1[ind][1])),
                 (int(pixelsImg2[ind][0] + w1), int(pixelsImg2[ind][1])), color)

    cv2.imwrite("matches/view" + str(idx) +".jpg", view)
    # cv2.waitKey(0)

def NormalizedFMatrix(points1,points2):

    dist1 = np.sqrt((points1[:,0]- np.mean(points1[:,0]))**2 + (points1[:,1]- np.mean(points1[:,1]))**2)
    dist2 = np.sqrt((points2[:,0]- np.mean(points2[:,0]))**2 + (points2[:,1]- np.mean(points2[:,1]))**2)

    m_dist1 = np.mean(dist1)
    m_dist2 = np.mean(dist2)

    scale1 = np.sqrt(2)/m_dist1
    scale2 = np.sqrt(2)/m_dist2

    t1 = np.array([[scale1, 0, -scale1*np.mean(points1[:,0])],[0, scale1, -scale1*np.mean(points1[:,1])],[0, 0, 1]])
    t2 = np.array([[scale2, 0, -scale2*np.mean(points2[:,0])],[0, scale2, -scale2*np.mean(points2[:,1])],[0, 0, 1]])


    U_x = (points1[:,0] - np.mean(points1[:,0]))*scale1
    U_y = (points1[:,1] - np.mean(points1[:,1]))*scale1
    V_x = (points2[:,0] - np.mean(points2[:,0]))*scale2
    V_y = (points2[:,1] - np.mean(points2[:,1]))*scale2

    A = np.zeros((len(U_x),9))

    for i in range(len(U_x)):

        A[i] = np.array([U_x[i]*V_x[i], U_y[i]*V_x[i], V_x[i], U_x[i]*V_y[i], U_y[i]*V_y[i], V_y[i], U_x[i], U_y[i], 1])

    U,S,V = np.linalg.svd(A)
    V = V.T
    F = V[:,-1].reshape(3,3)

    Uf, Sf, Vf = np.linalg.svd(F)
    SF = np.diag(Sf)
    SF[2,2] = 0

    F = Uf @ SF @ Vf
    F = t2.T @ F @ t1
    F = F/F[2,2]

    return F


if __name__ == '__main__':
    value = '/home/srujan/PycharmProjects/visual_odometry/stereo/centre'

    # global F

    # fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('./model')
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('/home/srujan/PycharmProjects/visual_odometry/model')


    # calibration matrix
    k= np.array([[fx,0,cx],
                [0,fy,cy],
                [0,0,1]])

    image_list = []

    filenames = [img for img in glob.glob("/home/srujan/PycharmProjects/visual_odometry/stereo/centre/*.png")]
    # filenames = [img for img in glob.glob("/home/srujan/PycharmProjects/visual_odometry/samples/*.png")]

    filenames.sort()
    for img in filenames:
        image_list.append(img)

    # image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.png')]

    Homogeneous_matrix_new  = np.eye(4,4)

    Trajectories = []

    frame_count = 0

    max_len = 250

    # for q in range(25,len(image_list)-1):
    for q in range(30,max_len):


        print('frame count before entering ',q)

        image1 = cv2.imread(image_list[q],0)
        image2 = cv2.imread(image_list[q+1],0)


        color_image1 = cv2.cvtColor(image1, cv2.COLOR_BayerGR2BGR)
        # color_image1 = cv2.cvtColor(image1, cv2.COLOR_BayerGR2GRAY)

        undistorted_image1 = UndistortImage.UndistortImage(color_image1, LUT)
        color_image2 = cv2.cvtColor(image2, cv2.COLOR_BayerGR2BGR)
        # color_image2 = cv2.cvtColor(image2, cv2.COLOR_BayerGR2GRAY)

        undistorted_image2 = UndistortImage.UndistortImage(color_image2, LUT)
        # cv2.imshow('undistorted', undistorted_image1)
        # cv2.waitKey(0)

        gray1 = cv2.cvtColor(undistorted_image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2GRAY)

        equ1 = cv2.equalizeHist(gray1)
        img1_gray = cv2.GaussianBlur(equ1, (3, 3), 0)
        equ2 = cv2.equalizeHist(gray2)
        img2_gray = cv2.GaussianBlur(equ2, (3, 3), 0)

        # Apply SIFT feature detector
        all_matches = sift_feat(undistorted_image1,undistorted_image2)

        ## Ransac:
        ULA = np.delete(all_matches[0], 2, 1)
        VLA = np.delete(all_matches[1], 2, 1)

        # E_, _ = cv2.findEssentialMat(ULA, VLA, cameraMatrix=k, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        # F_tilde_ = cv2.findFundamentalMat(ULA, VLA, method=cv2.RANSAC)

        inliers_,S_ = RANSAC(all_matches)

        S_array = np.asarray(S_)

        img1_inliers = np.vstack((S_array[:,:,0][:,0], S_array[:,:,1][:,0])).T
        img2_inliers = np.vstack((S_array[:,:,0][:,1], S_array[:,:,1][:,1])).T

        # vizMatches(undistorted_image1,undistorted_image2,img1_inliers,img2_inliers,q)

        # Estimate Essential matrix
        rand_points=random.sample(range(0,len(img1_inliers)),8)
        # x_i = list(itemgetter(*rand_points)(img1_inliers))
        # x_i_dash = list(itemgetter(*rand_points)(img2_inliers))

        x_i = np.asarray(list(itemgetter(*rand_points)(img1_inliers)))
        x_i_dash = np.asarray(list(itemgetter(*rand_points)(img2_inliers)))

        # F_tilde_ = NormalizedFMatrix(x_i, x_i_dash)

        F_tilde_ = estimate_fundamental_matrix(x_i, x_i_dash)

        print("F_tile", F_tilde_)

        E = obtain_essential_matrix(F_tilde_, k)
        #
        U,D,Vh = np.linalg.svd(E)
        W=np.array([[0,-1,0],[1,0,0],[0,0,1]])

        C1=U[:,2]
        C2=-U[:,2]
        C3=U[:,2]
        C4=-U[:,2]
        C_list = [C1, C2, C3, C4]

        R1 = np.matmul(U, np.matmul(W, Vh))
        R2 = np.matmul(U, np.matmul(W, Vh))
        R3 = np.matmul(U, np.matmul(W.T, Vh))
        R4 = np.matmul(U, np.matmul(W.T, Vh))
        R_list = [R1, R2, R3, R4]

        for i in range(0, len(C_list)):
            if det(R_list[i]) < 0:
                C_list[i] = -C_list[i]

        correct_poses = []

        prev = 0
        # for R,C in zip(R_list, C_list):
        #
        #
        #     values = []
        #     positive = 0
        #
        #     P1 = np.eye(3,4)
        #     P2 = np.hstack((R,np.reshape((C.T), (3,1))))
        #
        #     x_pts , ones_ = linear_LS_triangulation(img1_inliers, P1, img2_inliers, P2)
        #
        #     for i in range(0,len(img1_inliers)):
        #
        #         value = np.dot(np.reshape(R[:,2], (1,3)), (x_pts - C)[i])
        #         # values.append(value)
        #
        #         if value >0:
        #             positive+=1
        # #     print('number of positives', positive)
        # #     print('R', R)
        # #     print('C', C)
        #
        #     if positive > prev:
        #         correct_poses = [R,C]
        #         prev = positive
        #
        _, cur_R, cur_t, mask = cv2.recoverPose(E, ULA, VLA, cameraMatrix=k)

        if np.linalg.det(cur_R)<0:
            cur_R = -cur_R
            cur_t = -cur_t

        correct_poses = [cur_R,cur_t]

        Obtained_homogeneous_matrix = np.eye(4,4)
        Obtained_homogeneous_matrix[0:3, 0:3] = correct_poses[0]
        Obtained_homogeneous_matrix[0:3, 3:4] = np.reshape(correct_poses[1], (3,1))

        x_old = Homogeneous_matrix_new[0][3]
        y_old = Homogeneous_matrix_new[1][3]
        z_old = Homogeneous_matrix_new[2][3]

        x_new = Obtained_homogeneous_matrix[0][3]
        y_new = Obtained_homogeneous_matrix[1][3]
        z_new = Obtained_homogeneous_matrix[2][3]

        Homogeneous_matrix_new = np.matmul(Homogeneous_matrix_new, Obtained_homogeneous_matrix)

        # Trajectories.append([[x_old, y_old, z_old],[x_new, y_new, z_new]])
        Trajectories.append([x_old, y_old, z_old,x_new, y_new, z_new])

        q+=1

        print('frame count after exiting ',q)

    # np.save('trajectories_builtin_edited.npy', Trajectories)





