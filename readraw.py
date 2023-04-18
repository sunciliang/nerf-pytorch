import rawpy
from PIL import Image
import matplotlib.pyplot as plt
image_path = r'C:\Users\main\Desktop\tt\IMG_20230408_144836.dng'
image_path1 = r'C:\Users\main\Desktop\tt\IMG_20230408_144836.jpg'
img = Image.open(image_path1)

plt.figure(1)
plt.subplot(1, 2, 1)  # 图一包含1行2列子图，当前画在第一行第一列图上
plt.imshow(img)

def torgb():
    m, n, k = ryb.shape

    wb_rgb = np.array([wb[0], wb[1] - wb[0], wb[2]], dtype=np.float)

    rgb = ryb
    rgb[:, :, 1] = ryb[:, :, 1] - ryb[:, :, 0]
    rgb = rgb / np.tile(wb_rgb, (m, n, 1))

    rgb = np.maximum(0, rgb / np.max(rgb))

    return rgb

with rawpy.imread(image_path) as raw:
    rgb = raw.postprocess(user_wb=(1,1,1,1),four_color_rgb = False , no_auto_bright=True, output_bps=8, gamma=(1,1))
    plt.subplot(1, 2, 2)  # 当前画在第一行第2列图上
    plt.imshow(rgb)
    #use_camera_wb
    plt.show()

