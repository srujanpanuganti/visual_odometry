import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import cv2


gs = plt.GridSpec(2,3)



array6XN = np.load('/home/srujan/PycharmProjects/visual_odometry/trajectories.npy')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=(array6XN[:,3]), ys=0, zs=(array6XN[:,5] * -1), zdir='z', s=20, c=None)

x_vals = (array6XN[:,3]) + abs(np.amin(array6XN[:,3]))
z_vals = (array6XN[:,5]) + abs(np.amin(array6XN[:,5]))


ax.scatter(xs=x_vals, ys=0, zs=z_vals, zdir='z', s=20, c=None)

#
#
# for i in range(0,array6XN[:,3].shape[0]):
#     print("FRAME %d"%i)
#     j=i+30
#     # img=cv2.imread("FRAMES/%d.jpg"%j)
#     # img2=cv2.imread("SIFT/%d.jpg"%j)
#     # plt.plot(array6XN[i][3],array6XN[i][5],'o',color='blue')
#     ax.scatter(xs=array6XN[i][3], ys=0, zs=array6XN[i][5], zdir='z', s=20, c=None)
#
#     # plt.plot(x_new2[i],z_new2[i],'o',color='red')
#     # img=cv2.resize(img, (0,0), fx=0.3, fy=0.3)
#     # img2=cv2.resize(img2, (0,0), fx=0.6, fy=0.6)
#     # cv2.imshow("Original_Image",img)
#     # cv2.moveWindow("Original_Image",800,100)
#     # cv2.imshow("SIFT_Image",img2)
#     # cv2.moveWindow("SIFT_Image",800,450)
#     plt.pause(0.01)
#     # cv2.waitKey(1)


# 1399381511633263
