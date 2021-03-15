import cv2
import numpy as np
import torch
from tqdm import tqdm

import cfg
from datagen import get_input_data, pre_process_img
from model import Generator
from utils import save_result

device = torch.device('cuda:3')


def predict(i_t, i_s, model, to_shape=None):
    i_t, i_s, to_shape = pre_process_img(i_t, i_s, to_shape)

    o_sk, o_t, o_b, o_f = model([i_t.to(device), i_s.to(device)])

    o_sk = o_sk.data.cpu()
    o_t = o_t.data.cpu()
    o_b = o_b.data.cpu()
    o_f = o_f.data.cpu()

    transpose_vector = [0, 2, 3, 1]
    o_sk = o_sk.permute(transpose_vector).numpy()
    o_t = o_t.permute(transpose_vector).numpy()
    o_b = o_b.permute(transpose_vector).numpy()
    o_f = o_f.permute(transpose_vector).numpy()

    o_sk = cv2.resize((o_sk[0] * 255.).astype(np.uint8), to_shape, interpolation=cv2.INTER_NEAREST)
    o_t = cv2.resize(((o_t[0] + 1.) * 127.5).astype(np.uint8), to_shape)
    o_b = cv2.resize(((o_b[0] + 1.) * 127.5).astype(np.uint8), to_shape)
    o_f = cv2.resize(((o_f[0] + 1.) * 127.5).astype(np.uint8), to_shape)

    return [o_sk, o_t, o_b, o_f]


def load_model(model_path):
    G = Generator().to(device)

    # 多GPU机器训练的模型需要做map_location
    checkpoint = torch.load(model_path, map_location=device)
    G.load_state_dict(checkpoint)

    # 预测时应该调整为eval模式，否则图像质量非常差
    G.eval()
    return G


def main(date, number):
    model_path = 'model_logs/checkpoints/{}/iter-{}/G.pth'. \
        format(date, str(number).zfill(len(str(cfg.max_iter))))
    save_path = 'examples/results/{}/iter-{}'. \
        format(date, str(number).zfill(len(str(cfg.max_iter))))

    G = load_model(model_path)

    for data in get_input_data():
        i_t, i_s, original_shape, data_name = data
        result = predict(i_t, i_s, G, original_shape)
        save_result(save_path, result, data_name, mode=1)


if __name__ == '__main__':
    for i in tqdm(range(0, cfg.max_iter, 1000)):
        # main('20210310225258', i + 1000)
        main('20210315142514', i + 1000)
        if i + 1000 == 2000:
            break
