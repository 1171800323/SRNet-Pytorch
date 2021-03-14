import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cfg


class datagen_srnet(Dataset):
    def __init__(self):
        self.name_list = os.listdir(os.path.join(cfg.data_dir, cfg.t_b_dir))

    def __getitem__(self, item):
        name = self.name_list[item]

        i_t = cv2.imread(os.path.join(cfg.data_dir, cfg.i_t_dir, name))
        i_s = cv2.imread(os.path.join(cfg.data_dir, cfg.i_s_dir, name))
        t_sk = cv2.imread(os.path.join(cfg.data_dir, cfg.t_sk_dir, name), cv2.IMREAD_GRAYSCALE)
        t_t = cv2.imread(os.path.join(cfg.data_dir, cfg.t_t_dir, name))
        t_b = cv2.imread(os.path.join(cfg.data_dir, cfg.t_b_dir, name))
        t_f = cv2.imread(os.path.join(cfg.data_dir, cfg.t_f_dir, name))
        mask_t = cv2.imread(os.path.join(cfg.data_dir, cfg.mask_t_dir, name), cv2.IMREAD_GRAYSCALE)  # 读取单通道灰度图像

        return [i_t, i_s, t_sk, t_t, t_b, t_f, mask_t]

    def __len__(self):
        return len(self.name_list)


def collate_fn(batch):
    i_s_batch, i_t_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []

    # 对训练数据统一形状
    w_sum = 0
    for item in batch:
        t_b = item[4]
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)

    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_w, to_h)  # cv2.resize目标尺寸参数是：[w, h]

    for item in batch:
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = item
        i_t = cv2.resize(i_t, to_scale)
        i_s = cv2.resize(i_s, to_scale)
        t_sk = cv2.resize(t_sk, to_scale, interpolation=cv2.INTER_NEAREST)
        t_t = cv2.resize(t_t, to_scale)
        t_b = cv2.resize(t_b, to_scale)
        t_f = cv2.resize(t_f, to_scale)
        mask_t = cv2.resize(mask_t, to_scale, interpolation=cv2.INTER_NEAREST)  # 最近邻插值

        i_t_batch.append(i_t)
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t)
        t_b_batch.append(t_b)
        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)

    # 调整维度
    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.expand_dims(np.stack(t_sk_batch), axis=-1)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.expand_dims(np.stack(mask_t_batch), axis=-1)

    # 将像素值转换为[-1,1]的值，规范化，并转成Tensor
    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.)
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.)
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.)
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.)
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.)
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.)
    mask_t_batch = torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)

    transpose_vector = [0, 3, 1, 2]

    return [i_t_batch.permute(transpose_vector), i_s_batch.permute(transpose_vector),
            t_sk_batch.permute(transpose_vector),
            t_t_batch.permute(transpose_vector), t_b_batch.permute(transpose_vector),
            t_f_batch.permute(transpose_vector), mask_t_batch.permute(transpose_vector)]


# 读取输入数据，该数据为测试用例
def get_input_data(data_dir=cfg.example_data_dir):
    data_list = os.listdir(data_dir)
    data_list = [data_name.split('_')[0] + '_' for data_name in data_list]
    data_list = list(set(data_list))
    res_list = []

    transpose_vector = [0, 3, 1, 2]

    for data_name in data_list:
        i_t = cv2.imread(os.path.join(data_dir, data_name + 'i_t.png'))
        i_s = cv2.imread(os.path.join(data_dir, data_name + 'i_s.png'))

        h, w = i_t.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        to_h = cfg.data_shape[0]
        to_w = int(w * scale_ratio)
        to_w = int(round(to_w / 8)) * 8
        to_scale = (to_w, to_h)  # cv2.resize目标尺寸参数是：[w, h]
        i_t = cv2.resize(i_t, to_scale).astype(np.float32) / 127.5 - 1
        i_s = cv2.resize(i_s, to_scale).astype(np.float32) / 127.5 - 1
        i_t = torch.from_numpy(np.expand_dims(i_t, axis=0))
        i_s = torch.from_numpy(np.expand_dims(i_s, axis=0))

        res_list.append([i_t.permute(transpose_vector), i_s.permute(transpose_vector),
                         (w, h), data_name])  # cv2.resize目标尺寸参数是：[w, h]
    return res_list


def pre_process_img(i_t, i_s, to_shape):
    if len(i_t.shape) == 3:
        h, w = i_t.shape[:2]
        if not to_shape:
            to_shape = (w, h)  # w first for cv2
        if i_t.shape[0] != cfg.data_shape[0]:
            ratio = cfg.data_shape[0] / h
            predict_h = cfg.data_shape[0]
            predict_w = round(int(w * ratio) / 8) * 8
            predict_scale = (predict_w, predict_h)  # w first for cv2
            i_t = cv2.resize(i_t, predict_scale)
            i_s = cv2.resize(i_s, predict_scale)
        if i_t.dtype == np.uint8:
            i_t = i_t.astype(np.float32) / 127.5 - 1
            i_s = i_s.astype(np.float32) / 127.5 - 1
        i_t = torch.from_numpy(np.expand_dims(i_t, axis=0))
        i_s = torch.from_numpy(np.expand_dims(i_s, axis=0))

        transpose_vector = [0, 3, 1, 2]
        i_t = i_t.permute(transpose_vector)
        i_s = i_s.permute(transpose_vector)
    return i_t, i_s, to_shape


if __name__ == '__main__':
    # get_input_data()

    train_data = DataLoader(dataset=datagen_srnet(), batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True,
                            num_workers=4)
    train_data_iter = iter(train_data)
    for i in range(10):
        print(next(train_data_iter))
