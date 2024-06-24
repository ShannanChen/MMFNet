# import os
# import shutil
#
# # 设置你的nii文件夹的路径
# nii_dir = r'I:\Paper5\GCB-DIFF\2018\MICCAI2018_3DUXNET_seg_all_loss_embed\nii'
#
# # 遍历nii文件夹
# for root, dirs, files in os.walk(nii_dir, topdown=False):  # 注意：topdown设置为False以允许从子文件夹开始向上删除
#     for file in files:
#         if file.endswith('.nii.gz'):
#             # 构建文件的当前路径和目标路径
#             current_file_path = os.path.join(root, file)
#             target_file_path = os.path.join(nii_dir, file)
#
#             # 如果目标路径的文件已存在，先删除（小心使用！）
#             if os.path.exists(target_file_path):
#                 os.remove(target_file_path)
#
#             # 移动文件
#             shutil.move(current_file_path, target_file_path)
#             print(f'Moved: {current_file_path} -> {target_file_path}')
#
#     # 如果当前文件夹是空的，删除它
#     if root != nii_dir and not os.listdir(root):
#         os.rmdir(root)
#         print(f'Removed empty folder: {root}')
#
# print('Finished moving .nii.gz files and removing empty folders.')

from PIL import Image

# 读取图像
# image = Image.open(r'C:\Users\Administrator\Desktop\example_plot_binary.png')

# # 转换为灰度图像
# img_gray = image.convert('L')
#
# # 二值化处理
# img_binary = img_gray.point(lambda x: 255 if x > 127 else 0)
#
# # 保存为二值图像
# img_binary.save(r'C:\Users\Administrator\Desktop\filename_edge.png')


import numpy as np
from PIL import Image


# 加载图像并转换为numpy数组
def load_image(image_path):
    return np.array(Image.open(image_path))


# 将numpy数组保存为图像
def save_image(image_array, image_path):
    Image.fromarray(image_array).save(image_path)


# 向图像添加高斯噪声的函数
def add_gaussian_noise(image_array, noise_level):
    mean = 0
    sigma = noise_level
    gauss = np.random.normal(mean, sigma, image_array.shape).reshape(image_array.shape)
    noisy_array = image_array + gauss
    noisy_array = np.clip(noisy_array, 0, 255)  # 限制数值范围在0-255
    return noisy_array.astype('uint8')


# 主程序
def main():
    image_path = r'C:\Users\Administrator\Desktop\contours.png'  # 替换为你的图像路径
    image_array = load_image(image_path)

    # 定义保存图像的迭代次数
    save_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for i in range(1, 1001):
        # 添加噪声
        noise_level = 5  # 你可以根据需要调整噪声级别
        image_array = add_gaussian_noise(image_array, noise_level)

        # 如果当前迭代在保存步骤中，则保存图像
        if i in save_steps:
            save_image(image_array, f'noisy_image_{i}.png')


if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
#
# # 读取图像
# image_path = r'C:\Users\Administrator\Desktop\2.png'  # 替换为你的图像路径
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# # 检查图像是否被正确加载
# if image is not None:
#     # 应用二值化阈值处理
#     _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#
#     # 查找轮廓
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 创建全黑背景
#     contour_img = np.zeros_like(image)
#
#     # 绘制轮廓（边界）
#     cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)  # 白色边界，宽度为1个像素
#
#     # 保存提取的轮廓图像
#     cv2.imwrite('contours.png', contour_img)
#     print("Contours have been saved as 'contours.png'.")
# else:
#     print("Image not loaded correctly!")