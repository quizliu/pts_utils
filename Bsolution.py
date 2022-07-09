import os
import os.path as osp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
fps = 30
def reorder_vertices(vertices):
    # vertices.sort(key=lambda x: (x[1], x[0]))
    pass


# src_vertices = [[0.339583, 0.593935], [0.442500, 0.592456], [0.448750, 0.548077], [0.365000, 0.547337]]  # field left box
src_vertices = [[0.308854, 0.604167], [0.458073, 0.601389], [0.465625, 0.485648], [0.346615, 0.490278]]  # left box
# src_vertices = [[0.450521, 0.809259], [0.633854, 0.812963], [0.591406 ,0.609722], [0.457552, 0.611574]]  # mid box
img_x, img_y = 1920, 1080
src_vertices = np.array([[x * img_x, y * img_y] for x, y in src_vertices], dtype=np.float32)

x, y = 930, 1000
dx = 40
dy = 1.5 * dx 
dst_vertices = np.array([[x, y], [x + dx, y], [x + dx, y - dy], [x, y - dy]], dtype=np.float32)
rlpp = 4 / 5280 / dx  # 4 feet to miles

M = cv.getPerspectiveTransform(src_vertices, dst_vertices)
src_img = cv.imread('calibration_speed_53.jpg')
dst_img = cv.warpPerspective(src_img, M, (img_x, img_y))
cv.imwrite('after2.jpeg', dst_img)

# load detection data
data = []
with open('velocity_data/label_data_speed_53_1.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        data.append([float(line[1])] + [float(line[2])] +[int(line[-1])])
data.sort(key=lambda x: x[2])

fig = plt.figure(figsize=(16, 9))
plt.subplot(2,2,1)
for x, y, frame in data:
    plt.scatter(img_x * float(x), img_y * float(y))
plt.xlim([0, img_x])
plt.ylim([img_y, 0])

ax1 = fig.add_subplot(2,2,2)
plt.imshow(cv.cvtColor(src_img, cv.COLOR_BGR2RGB))
for x, y, frame in data:
    plt.scatter(img_x * float(x), img_y * float(y), c='r', s=1)

plt.subplot(2,2,3)
plt.imshow(cv.cvtColor(dst_img, cv.COLOR_BGR2RGB))

plt.subplot(2,2,4)
plt.xlim([0, img_x])
plt.ylim([img_y, 0])
new_data = []
for x, y, frame in data:
    src = np.array([[img_x * float(x)], [img_y * float(y)], [1]])
    dst = np.matmul(M, src)
    a, b, c = dst[0][0], dst[1][0], dst[2][0]
    new_data.append([a, b, frame])

def dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def time(f1, f2):
    return abs(f1 - f2) * 1 / fps

def velocity(pixel_d, pixel_t):
    return pixel_d * rlpp / pixel_t * 3600  # mph

for (*p1, f1), (*p2, f2) in zip(new_data, new_data[1:]):
    d = dis(p1, p2)
    t = time(f1, f2)
    v = velocity(d, t)
    print(v)

# plt.show()