import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import cv2

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
gs = plt.GridSpec(2,3)

# array6XN = np.load('/home/srujan/PycharmProjects/visual_odometry/src/trajectories_builtin_edited.npy')

# array6XN = np.load('/home/srujan/PycharmProjects/visual_odometry/src/trajectories_builtin_short.npy')


array6XN = np.load('/home/srujan/PycharmProjects/visual_odometry/src/trajectories_builtin_drone_full.npy')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')



# for i in range(0, array6XN[:,0].shape[0] -1):
#     ax.scatter(xs=(array6XN[:,0][i]), ys=(array6XN[:,1][i]), zs=(array6XN[:,2][i] * -1), zdir='z', s=20, c=None)
#     plt.savefig('/home/srujan/PycharmProjects/visual_odometry/drone3/pic{:>05}.jpg'.format(i))


# ax.scatter(xs=(array6XN[:,0] *-1), ys=(array6XN[:,1] *-1), zs=(array6XN[:,2] *-1), zdir='z', s=20, c=None)

# ax.scatter(xs=(array6XN[:,3] *-1), ys=0, zs=(array6XN[:,5] *-1), zdir='z', s=20, c=None)


# ani =[]
# def update(i):
#     ani.append(ax.scatter(xs=(array6XN[:,0][i] *-1), ys=(array6XN[:,1][i] *-1), zs=(array6XN[:,2][i] *-1), zdir='z', s=20, c=None))
    # ani.append(ax.scatter(xs=(array6XN[:,0][i] *-1), ys=(array6XN[:,1][i] *-1), zs=(array6XN[:,2][i] ), zdir='z', s=20, c=None))

# anim = animation.FuncAnimation(
#     fig, update, interval=100, frames=array6XN[:,0].shape[0])
# im_ani = animation.ArtistAnimation(fig, ani)
# print(len(ani))

# anim.save('drone_odom.mp4')

# im_ani.save('im.mp4', metadata={'artist':'Guido'})
'''
# myplot = ax.scatter(xs=(array6XN[:,0] *-1), ys=(array6XN[:,1] *-1), zs=(array6XN[:,2] *-1), zdir='z', s=20, c=None)

# line_ani = animation.FuncAnimation(
#     fig, myplot, 25, interval=50)
#
# plt.show()

# x_vals = (array6XN[:,3]) + abs(np.amin(array6XN[:,3]))
# z_vals = (array6XN[:,5]) + abs(np.amin(array6XN[:,5]))


# ax.scatter(xs=x_vals, ys=0, zs=z_vals, zdir='z', s=20, c=None)

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
'''

import cv2
# import numpy as np
import glob

file_names = []

img_array = []
for filename in glob.glob("/home/srujan/Downloads/MidAir/Kite_training/sunny/color_right/trajectory_0000/frames/*.JPEG"):
    # print("file_name : ", filename)
    file_names.append(filename)

print("num fo files ",len(file_names))
# print(ksjzchzb)

file_names.sort()

for file in file_names:

# for file in file_names[1000:2000]:
    # print("file_name : ", file)
    img = cv2.imread(file)
    # print("size is ", size)
    rez  = cv2.resize(img,(512, 512))
    height, width, layers = rez.shape
    size = (width,height)
    # print("resize is ", size)
    # print(xjgzh)
    img_array.append(rez)

print("length is ",len(img_array))
print("appended")

out = cv2.VideoWriter('drone_video0.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

print("now generating the video")
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("video generated")


