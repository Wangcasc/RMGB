import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import shutil
import json

ill_list = [
    '1_1_0_2_6',
    '1_1_105_0_2',
    '1_1_108_5_3',
    '1_1_108_5_7',
    '1_1_114_2_5',
    '1_1_116_0_5',
    '1_1_11_0_3',
    '1_1_11_0_9',
    '1_1_128_4_4',
    '1_1_134_2_2',
    '1_1_27_0_5',
    '1_1_39_0_2',
    '1_1_51_0_2',
    '1_1_53_0_2',
    '1_1_67_0_0',
    '1_1_74_0_2',
    '1_1_83_5_4',
    '1_1_93_0_9',
    '1_1_93_1_2',
    '2_1_30_2_0',
    '2_1_34_5_1',
    '2_1_3_3_7',
    '2_1_43_2_3',
    '2_1_47_1_7',
    '2_2_19_0_1',
    '2_2_26_2_4',
    # 上面是存在关键点缺失的样本，下面是关键点坐标超出图像边界的样本
    '1_1_104_0_0',
    '1_1_104_0_1',
    '1_1_104_0_2',
    '1_1_104_0_4',
    '1_1_104_0_5',
    '1_1_11_0_0',
    '1_1_121_0_5',
    '1_1_121_0_6',
    '1_1_121_0_7',
    '1_1_121_0_8',
    '1_1_121_0_9',
    '1_1_126_0_4',
    '1_1_128_0_4',
    '1_1_134_0_5',
    '1_1_134_0_6',
    '1_1_134_0_7',
    '1_1_134_0_8',
    '1_1_134_0_9',
    '1_1_137_0_8',
    '1_1_138_0_0',
    '1_1_138_0_1',
    '1_1_138_0_3',
    '1_1_139_0_6',
    '1_1_15_0_5',
    '1_1_1_0_5',
    '1_1_27_0_8',
    '1_1_28_0_1',
    '1_1_28_0_4',
    '1_1_32_0_4',
    '1_1_32_0_5',
    '1_1_32_0_6',
    '1_1_32_0_8',
    '1_1_32_0_9',
    '1_1_39_0_3',
    '1_1_39_0_6',
    '1_1_39_0_8',
    '1_1_39_0_9',
    '1_1_40_0_3',
    '1_1_41_0_6',
    '1_1_45_0_2',
    '1_1_45_0_8',
    '1_1_4_0_7',
    '1_1_4_0_8',
    '1_1_54_0_0',
    '1_1_54_0_1',
    '1_1_54_0_2',
    '1_1_54_0_3',
    '1_1_54_0_6',
    '1_1_54_0_7',
    '1_1_54_0_8',
    '1_1_54_0_9',
    '1_1_57_0_0',
    '1_1_57_0_4',
    '1_1_58_0_1',
    '1_1_58_0_5',
    '1_1_58_0_6',
    '1_1_58_0_7',
    '1_1_58_0_8',
    '1_1_67_0_7',
    '1_1_68_0_6',
    '1_1_68_0_7',
    '1_1_68_0_8',
    '1_1_69_0_4',
    '1_1_78_0_4',
    '1_1_78_0_5',
    '1_1_78_0_6',
    '1_1_78_0_7',
    '1_1_78_0_9',
    '1_1_80_0_6',
    '1_1_84_0_8',
    '1_1_86_0_3',
    '1_1_92_0_1',
    '1_1_93_0_4',
    '1_1_94_0_7',
    '1_1_99_0_3',
    '1_1_99_0_5',
    '1_1_99_0_7',
    '1_1_99_0_8',
    '1_1_99_0_9',
    '1_1_9_0_2',
    '1_1_9_0_7',
    '1_1_9_0_9',
    '2_1_15_0_3',
    '2_1_16_0_6',
    '2_1_27_0_5',
    '2_1_30_0_4',
    '2_1_30_0_8',
    '2_1_34_0_9',
    '2_1_40_0_0',
    '2_1_40_0_1',
    '2_1_40_0_2',
    '2_1_40_0_4',
    '2_1_40_0_9',
    '2_1_43_0_9',
    '2_1_47_0_3',
    '2_1_47_0_4',
    '2_1_47_0_5',
    '2_1_47_0_8',
    '2_1_4_0_1',
    '2_1_7_0_1',
    '2_1_7_0_2',
    '2_1_7_0_3',
    '2_2_20_0_6',
    '2_2_26_0_1',
    '2_2_27_0_7',
    '2_2_27_0_8',
    '2_2_28_0_2',
    '2_2_28_0_6',
    '2_2_28_0_7',
    '2_2_40_0_2',
    '2_2_9_0_6',
]

euclid_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

def get_mask(depth_img):
    # temporally used for Real_DHGA dataset mask generation
    ret, dst = cv2.threshold(depth_img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 找到最大区域并填充
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    if area:
        max_idx = np.argmax(area)
        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(dst, [contours[k]], 0)
    kernel1 = np.ones((3, 3), np.uint8)
    closed = cv2.erode(cv2.dilate(np.array(dst, dtype=np.uint8), kernel1), kernel1)
    # plt.imshow(closed)
    # plt.show()
    return closed

def get_avg_brightness(video_dir, mask_dir, video_list=None):
    total_brightness = []
    for video in video_list[:]:
        if video in ill_list:
            continue
        video_brightness = []
        video_path = os.path.join(video_dir, video)
        mask_path = os.path.join(mask_dir, video)
        img_list = os.listdir(video_path)
        for i in img_list:
            img = np.array(Image.open(os.path.join(video_path, i)).convert('RGB'))
            mask = np.array(Image.open(os.path.join(mask_path, i.replace('.jpg', '.png'))).convert('RGB'))
            mask = cv2.resize(mask, (200, 200))
            brightness = np.mean(img[mask>0])
            video_brightness.append(brightness)
        video_brightness = np.mean(video_brightness)
        total_brightness.append(video_brightness)
    avg_brightness = np.mean(total_brightness)
    print(avg_brightness)  # 120

def vis_pose(kpt: np.ndarray, img=None, save_name=None):
    '''
    Input:
    :param kpt: keypoint array with shape [21,2]
    :param img: image
    :param save_name: wether save the picture
    :param show: wether show the plt figure
    :return:
    '''
    color_lst = ["yellow", "blue", "green", "cyan", "magenta"]
    group_lst = [1, 5, 9, 13, 17, 21]
    plt.figure()
    if img is not None:
        plt.imshow(img*255)
    for j, color in enumerate(color_lst):
        x = np.insert(kpt[group_lst[j]:group_lst[j + 1], 0], 0, kpt[0, 0])
        y = np.insert(kpt[group_lst[j]:group_lst[j + 1], 1], 0, kpt[0, 1])
        plt.plot(x, y, color=color, linewidth=3, marker=".", markerfacecolor="red", markersize=15, markeredgecolor="red")
    plt.axis('off')
    plt.xlim((0, 200))
    plt.ylim((200, 0))
    if save_name is not None:
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        plt.savefig(save_name, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()

class BILASnorm:
    def __init__(self, video_dir, mask_dir, kpt_dir, save_dir=None):
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.kpt_dir = kpt_dir
        self.save_dir = save_dir
        self.video_list = os.listdir(video_dir)
        self.video_list.sort()

    def get_img(self, video_dir, video, image, transfer=False):
        # rewrite this function according to your situation
        # if you need to replace the suffix from jpg to png, set the transfer as True
        image = image.replace('jpg', 'png') if transfer else image
        img_path = os.path.join(os.path.join(video_dir, video), image)
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.resize(img, (200, 200))
        return img

    def get_kpt(self, video, image, max=None):
        # rewrite this function according to your situation
        kpt_path = os.path.join(self.kpt_dir, video + '.json')
        with open(kpt_path) as f:
            data = json.load(f)
            kpt_frames = data['info']
        kpt = np.array(kpt_frames[image]['keypoints'])
        if max:
            kpt = kpt * 200 // max
        return kpt

    def get_kpt_stru(self, video):
        kpt_path = os.path.join(self.kpt_dir, video + '.json')
        with open(kpt_path) as f:
            data = json.load(f)
        return data

    def background_norm(self, img, mask, show=False, save=False, video_name=None, image_name=None):
        mask[mask <= 200] = 0
        mask[mask > 200] = 1
        bg_norm = img * mask
        if show:
            plt.imshow(bg_norm)
            plt.show()
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'color_masked'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), bg_norm)
        return bg_norm

    def illumination_norm(self, img, mask, target_brightness, show=False, save=False, video_name=None, image_name=None):
        brightness = np.mean(img[mask > 0])
        dif = target_brightness - brightness
        img[mask > 0] = img[mask > 0] + dif
        if show:
            plt.imshow(img)
            plt.show()
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'light_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), img)
        return img

    def location_norm(self, img, kpt, show=False, save=False, video_name=None, image_name=None):
        # 先平移，将根节点放在（100,190)
        root_coord = kpt[0, :]
        bias = root_coord - np.array((100, 190))
        M = np.float32([[1, 0, -bias[0]], [0, 1, -bias[1]]])
        shifted = cv2.warpAffine(img, M, (200, 200))
        kpt_shifted = kpt - bias
        if show:
            vis_pose(kpt_shifted, shifted)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'loc_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), shifted)
        return shifted, kpt_shifted

    def angle_norm(self, img, kpt, show=False, save=False, video_name=None, image_name=None):
        # 根据中指与掌根连线的角度，将手掌摆正
        root_coord = kpt[0, :]
        mid_coord = kpt[9, :]
        tan = (mid_coord[0] - root_coord[0]) / (root_coord[1] - mid_coord[1])
        inv = np.degrees(np.arctan(tan))
        # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子,可以通过设置旋转中心，缩放因子以及窗口大小来防止旋转后超出边界的问题
        M = cv2.getRotationMatrix2D((int(root_coord[0]), int(root_coord[1])), inv, 1)
        rotated = cv2.warpAffine(img, M, (200, 200), borderValue=(0, 0, 0))  # M为上面的旋转矩阵, borderValue为空白区域的填充色
        M = cv2.getRotationMatrix2D((0, 0), -inv, 1)
        kpt = np.matmul(kpt, M[:, :2])
        kpt_rotated = kpt - (kpt[0, :] - root_coord)
        if show:
            vis_pose(kpt_rotated, rotated)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'angle_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), rotated)
        return rotated, kpt_rotated

    def scale_norm(self, img, kpt, show=False, save=False, video_name=None, image_name=None):
        # 将中指指根与掌根的连线长度归一化为统一大小
        sta = 80.0  # 手掌长度的标准大小
        root_coord = kpt[0, :]
        mid_coord = kpt[9, :]
        palm_len = euclid_distance(root_coord, mid_coord)
        scale_factor = palm_len / sta
        scaled = cv2.resize(img, (int(img.shape[1] / scale_factor), int(img.shape[0] / scale_factor)))
        kpt = (kpt / scale_factor).astype(np.int32)
        x, y = scaled.shape[0:2]
        if scale_factor > 1:
            scaled_final = np.zeros_like(img, dtype=np.uint8)
            scaled_final[190 - kpt[0, 1]:190 + y - kpt[0, 1], 100 - int(0.5 * x):100 + (x - int(0.5 * x)), :] = scaled
            kpt_scaled = kpt + (100 - int(0.5 * x), 190 - kpt[0, 1])
        elif scale_factor < 1:
            scaled_final = scaled[kpt[0, 1] - 190: kpt[0, 1] + 10, int(x * 0.5) - 100: int(x * 0.5) + 100, :]
            kpt_scaled = kpt - (int(0.5 * x) - 100, kpt[0, 1] - 190)
        else:
            scaled_final = scaled
            kpt_scaled = kpt
        if show:
            vis_pose(kpt_scaled, scaled_final)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'scale_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), scaled_final)
        return scaled_final, kpt_scaled

    def BILASnorm(self, img, mask, kpt):
        bg_norm = self.background_norm(img, mask, save=False)
        light_norm = self.illumination_norm(bg_norm, mask, 120, save=False, show=False)
        loc_norm, kpt_norm = self.location_norm(light_norm, kpt, save=False, show=False)
        ang_norm, kpt_norm = self.angle_norm(loc_norm, kpt_norm, save=False, show=False)
        sca_norm, kpt_norm = self.scale_norm(ang_norm, kpt_norm, save=False)
        return sca_norm, kpt_norm
        # return

    def process(self, video_list=None):
        if video_list == None:
            video_list = self.video_list
        for video in video_list:
            if video in ill_list:
                continue
            print(video)
            self.video_name = video
            video_path = os.path.join(video_dir, video)
            img_list = os.listdir(video_path)
            img_list.sort()
            kpt_dict = self.get_kpt_stru(video)
            for im_name in img_list:
                self.image_name = im_name
                img = self.get_img(self.video_dir, video, im_name)
                mask = self.get_img(self.mask_dir, video, im_name)
                kpt = self.get_kpt(video, im_name, max=480)
                # BILASnorm
                sca_norm, kpt_norm = self.BILASnorm(img, mask, kpt)
                # vis_pose(kpt_norm, sca_norm)
                save_dir = os.path.join(os.path.join(self.save_dir, 'color_norm'), self.video_name)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                imageio.imwrite(os.path.join(save_dir, self.image_name), sca_norm)
                kpt_dict['info'][im_name]['keypoints'] = kpt.tolist()
            kpt_dict['maker'] = 'Real-DHGA'
            kpt_save_path = os.path.join(os.path.join(self.save_dir, 'keypoints_v1_norm'), video+'.json')
            with open(kpt_save_path, 'w') as file:
                json.dump(kpt_dict, file)

    def process_bg_norm(self, video_list=None):
        '''
        for raw DHGA background norm process, not DHGA-br
        :param video_list:
        :return:
        '''
        if video_list == None:
            video_list = self.video_list
        for video in video_list:
            if video in ill_list:
                continue
            print(video)
            video_path = os.path.join(video_dir, video)
            img_list = os.listdir(video_path)
            img_list.sort()
            for im_name in img_list:
                img = self.get_img(self.video_dir, video, im_name)
                mask = self.get_img(self.mask_dir, video, im_name.replace('jpg', 'png'))
                # kpt = self.get_kpt(video, im_name)
                # BILASnorm
                self.background_norm(img, mask, save=True, video_name=video, image_name=im_name)



if __name__ == '__main__':
    # video_dir = r'D:\ZYF\datasets\DHG-Auth/color_hand'
    # mask_dir = r'D:\ZYF\datasets\DHG-Auth/mask'
    # kpt_dir = r'D:\ZYF\datasets\DHG-Auth/keypoints_wxl'
    # save_dir = r'D:\ZYF\datasets\DHG-Auth'
    video_dir = r'D:\ZYF\datasets\RealDHGA/color'
    mask_dir = r'D:\ZYF\datasets\RealDHGA/mask'
    kpt_dir = r'D:\ZYF\datasets\RealDHGA/keypoints_v1'
    save_dir = r'D:\ZYF\datasets\RealDHGA'
    bilasNorm = BILASnorm(video_dir, mask_dir, kpt_dir, save_dir=save_dir)
    img_norm = bilasNorm.process(['1_1_10_10_1'])


