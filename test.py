import glob
import os, imageio
from matplotlib import pyplot as plt
import numpy as np
import math
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
import rawpy
import copy

def convert_K_to_RGB(colour_temperature):
    """
    Converts from K to RGB, algorithm courtesy of
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    # range check
    if colour_temperature < 1000:
        colour_temperature = 1000
    elif colour_temperature > 40000:
        colour_temperature = 40000

    tmp_internal = colour_temperature / 100.0

    # red
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red

    # green
    if tmp_internal <= 66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green

    # blue
    if tmp_internal >= 66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue

    return red, green, blue

# imgdir = r'E:\GitHub\nerf-py\data\t'
# savedir = r'E:\GitHub\nerf-py\data\test_synthetic\input'
# imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
# def imread(f):
#     if f.endswith('png'):
#         return imageio.imread(f, ignoregamma=True)
#     else:
#         return imageio.imread(f)
# imgs = [imread(f)[...,:3]/255. for f in imgfiles]
#
# K = [2700,3000,5500,9000,14000,34000]
# print('dd')
# for idx, i in enumerate(imgs):
#     # cropped_img = i[700:4700, 400:6800]
#     for idxtt, temps in enumerate(K):
#         print(temps)
#         temp = copy.deepcopy(i)
#         temp = temp.reshape(-1,3)
#         r,g,b = convert_K_to_RGB(temps)
#         r_adjust, g_adjust, b_adjust = r / g, g / g, b / g
#         adjust = np.array([])
#         temp[:, 0] = temp[:, 0] * r_adjust
#         temp[:, 1] = temp[:, 1] * g_adjust
#         temp[:, 2] = temp[:, 2] * b_adjust
#         temp = temp.reshape(i.shape[0], i.shape[1], 3)
#
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 2, 1)
#         imgplot = plt.imshow(i)
#         ax.set_title('Before')
#         ax = fig.add_subplot(1, 2, 2)
#         imgplot = plt.imshow(temp)
#         ax.set_title('After')
#
#
#         plt.show()
#
#         # rgb8 = to8b(temp)
#         # filename = os.path.join(savedir, '{:03d}{}.png'.format(idx,idxtt))
#         # imageio.imwrite(filename, temp)
#         # print(idx)



import numpy as np
tt = []
# key = [5000,5000,5000,5000,5000]
# key = [2.0,1.0,0.5,0.25,0.125]
key = [1.0,1.0,1.0,1.0,1.0]
for i in range(80):
    tt.append(key[i%5])
#np.save('temperature.npy',tt)
np.save('exposure.npy',tt)
# #
# k=np.load(r'E:\GitHub\nerf-py\temperature.npy')
# print(k)

# path = glob.glob(r'E:\GitHub\nerf-py\data\colorboard18\RAW\*.dng')
# for idx,img in enumerate(path):
#     with rawpy.imread(img) as raw:
#             rgb = raw.postprocess(gamma=(1,1), use_auto_wb=False,use_camera_wb =True, output_bps=16)
#     imageio.imsave(r'E:\GitHub\nerf-py\data\colorboard18\images\{:03d}.jpg'.format(idx), rgb)