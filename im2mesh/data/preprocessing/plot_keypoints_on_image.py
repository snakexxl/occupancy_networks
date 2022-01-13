import numpy as np
from PIL import Image
from matplotlib import pylab as plt


def plot_keypoints_on_image(image_path: str, poseX, poseY):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    xy2imgxy = lambda x, y: (img.size[0] * x / np.max(ticklx),
                             img.size[1] * (np.max(tickly) - y) / np.max(tickly))
    ticklx = np.linspace(0, 1002, 6)
    tickly = np.linspace(0, 1000, 6)
    tickpx, tickpy = xy2imgxy(ticklx, tickly)
    # Rewrite x,y ticks
    ax.set_xticks(tickpx)
    ax.set_yticks(tickpy)
    ax.set_xticklabels(ticklx.astype('int'))
    ax.set_yticklabels(tickly.astype('int'))
    # Add scatter point on the image.
    px = poseX.cpu()
    # px = 1002-px
    py = poseY.cpu()
    py = 1000 - py
    imgx, imgy = xy2imgxy(px, py)
    ax.scatter(imgx, imgy, s=10, lw=3, facecolor="none", edgecolor="red")

    # ax.invert_yaxis()
    def connectpoints(imgx, imgy, p1, p2):
        x1, x2 = imgx[p1], imgx[p2]
        y1, y2 = imgy[p1], imgy[p2]
        l1, = plt.plot([x1, x2], [y1, y2], 'k-')
        l1.set_color('b')

    #lower body part
    connectpoints(imgx, imgy, 0, 1)
    connectpoints(imgx, imgy, 1, 2)
    connectpoints(imgx, imgy, 2, 3)
    connectpoints(imgx, imgy, 3, 4)
    connectpoints(imgx, imgy, 4, 5)
    connectpoints(imgx, imgy, 0, 6)
    connectpoints(imgx, imgy, 6, 7)
    connectpoints(imgx, imgy, 7, 8)
    connectpoints(imgx, imgy, 8, 9)
    connectpoints(imgx, imgy, 9, 10)
    #upper body part
    connectpoints(imgx, imgy, 0, 11)
    connectpoints(imgx, imgy, 11, 12)
    connectpoints(imgx, imgy, 12, 13)
    #arm
    connectpoints(imgx, imgy, 13, 17)
    connectpoints(imgx, imgy, 13, 25)
    connectpoints(imgx, imgy, 17, 18)
    connectpoints(imgx, imgy, 18, 19)
    connectpoints(imgx, imgy, 19, 21)
    connectpoints(imgx, imgy, 19, 22)
    connectpoints(imgx, imgy, 25, 26)
    connectpoints(imgx, imgy, 26, 27)
    connectpoints(imgx, imgy, 27, 30)
    connectpoints(imgx, imgy, 27, 29)
    # ##head
    connectpoints(imgx, imgy, 13, 14)
    connectpoints(imgx, imgy, 14, 15)
    #punkte die wegen veranschaulichung nicht angezeigt werden.
    #16,20,23,28,31 sind bei den Händen und einer beim Hals
    # connectpoints(imgx, imgy, 16, 20)
    # connectpoints(imgx, imgy, 20, 23)
    # connectpoints(imgx, imgy, 23, 24)
    # connectpoints(imgx, imgy, 28, 31)
    # connectpoints(imgx, imgy, 31, 16)
    plt.xlabel("X-Werte")
    plt.ylabel("Y-Werte")
    plt.show()