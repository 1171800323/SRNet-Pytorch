import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import cfg

PrintColor = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'amaranth': 35,
    'ultramarine': 36,
    'white': 37
}

PrintStyle = {
    'default': 0,
    'highlight': 1,
    'underline': 4,
    'flicker': 5,
    'inverse': 7,
    'invisible': 8
}


def get_train_name():
    # get current time for train name
    return datetime.now().strftime('%Y%m%d%H%M%S')


def print_log(s, time_style=PrintStyle['default'], time_color=PrintColor['blue'],
              content_style=PrintStyle['default'], content_color=PrintColor['white']):
    # colorful print s with time log
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print(log)


def save_result(save_dir, result, name, mode):
    # 保存输出图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if mode == 1:
        o_sk, o_t, o_b, o_f = result
        cv2.imwrite(os.path.join(save_dir, name + 'o_f.png'), o_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_sk.png'), o_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_t.png'), o_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_b.png'), o_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


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


def get_log_writer(train_name):
    if sys.platform.startswith('win'):
        writer = SummaryWriter('model_logs\\train_logs\\' + train_name)
    else:
        writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, train_name))
    return writer
