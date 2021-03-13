import os
import sys
from datetime import datetime

import cv2
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
    o_sk, o_t, o_b, o_f = result
    cv2.imwrite(os.path.join(save_dir, name + 'o_f.png'), o_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    if mode == 1:
        cv2.imwrite(os.path.join(save_dir, name + 'o_sk.png'), o_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_t.png'), o_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_b.png'), o_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def get_log_writer(train_name):
    if sys.platform.startswith('win'):
        writer = SummaryWriter('model_logs\\train_logs\\' + train_name)
    else:
        writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, train_name))
    return writer
