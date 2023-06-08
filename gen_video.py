import cv2
import os

# 获取图片目录中所有图片的路径
img_dir = r'C:\Users\main\Desktop\fsdownload\temp\test_preds_T5200.0_E0_A0.02/'
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

# 获取第一张图片的尺寸
img = cv2.imread(img_paths[0])
height, width, _ = img.shape

# 定义新视频的输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'C:\Users\main\Desktop\fsdownload\temp\test_preds_T5200.0_E0_A0.02\new_video.mp4', fourcc, 30, (width, height))

# 将每张图片写入新视频
for img_path in img_paths:
    img = cv2.imread(img_path)

    # 写入新视频
    out.write(img)

# 释放资源
out.release()