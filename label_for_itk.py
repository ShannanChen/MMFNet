# from lungmask import mask  #lungmask refer:https://github.com/JoHof/lungmask
import SimpleITK as sitk
import os
from tqdm import trange

def get_ct_file(main_path):
    ctpath = []
    ct_list = os.listdir(main_path)  # 列出文件夹下所有的目录与文件
    # 遍历该文件夹下的所有目录或者文件
    for ii in range(0, len(ct_list)):
        path = os.path.join(main_path, ct_list[ii])
        if path.endswith('.gz'):
            ctpath.append(path)
    return ctpath
# def get_ct_file(main_path):
#     ctpath = []
#     # 遍历该文件夹下的所有目录或者文件
#     for root, s_dirs, _ in os.walk(main_path, topdown=True):  # 获取 train文件下各文件夹名称
#         for sub_dir in s_dirs:
#             i_dir = os.path.join(root, sub_dir) # 获取各类的文件夹 绝对路径
#             img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
#             for i in range(len(img_list)):
#                 if img_list[i].endswith('.gz'):
#                     path = os.path.join(i_dir, img_list[i])
#                     ctpath.append(path)
    # return ctpath
if __name__ == '__main__':

    gt_path = r'I:\Paper5\GCB-DIFF\2018\labelsVal'
    prediction_path = r'I:\Paper5\GCB-DIFF\2018\MICCAI2018_SwinUNETR_seg_all_loss_embed\nii'
    # save_path = r'I:\mip_paper\2_lung_mask\HC'
    gt_path = get_ct_file(gt_path)
    gt_path.sort()
    predict_path = get_ct_file(prediction_path)
    predict_path.sort()

    for i in trange(len(gt_path)):
        gt_image = sitk.ReadImage(gt_path[i])
        gt_image_arr = sitk.GetArrayFromImage(gt_image)
        predict_image = sitk.ReadImage(predict_path[i])
        predict_image_arr = sitk.GetArrayFromImage(predict_image)
        inter_img = predict_image_arr - gt_image_arr
        inter_img[inter_img==1] = 3 #过分割
        # inter_img[inter_img>3] = 2
        inter_img[inter_img<0] = 2  #欠分割
        inter_img2 = predict_image_arr + gt_image_arr
        inter_img[inter_img2==2] = 1 #分割正确
        # inter_img = inter_img + gt_image_arr # 1：GT 正确分割
        # new_inter_img = inter_img + gt_image_arr
        new_img = sitk.GetImageFromArray(inter_img)
        new_img.SetDirection(gt_image.GetDirection())
        new_img.SetOrigin(gt_image.GetOrigin())
        new_img.SetSpacing(gt_image.GetSpacing())
        # new_file_path = save_path
        # path = os.path.join(new_file_path, img)
        save_path = predict_path[i].replace('new','itk')
        sitk.WriteImage(new_img, save_path)
