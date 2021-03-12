import cv2
import numpy as np
import torch

from datagen import get_input_data
from model import Generator
from utils import pre_process_img, save_result

device = torch.device('cuda:3')


def predict(i_t, i_s, to_shape=None):
    i_t, i_s, to_shape = pre_process_img(i_t, i_s, to_shape)

    G = Generator().to(device)
    checkpoint = torch.load("model_logs/checkpoints/20210310225258/iter-068000/G.pth")

    # 多GPU机器训练的模型需要做map_location
    # checkpoint = torch.load("model_logs/checkpoints/G.pth", map_location='cuda:0')

    G.load_state_dict(checkpoint)
    G.eval()
    o_sk, o_t, o_b, o_f = G([i_t.to(device), i_s.to(device)])

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


if __name__ == '__main__':
    for data in get_input_data():
        i_t, i_s, original_shape, data_name = data
        result = predict(i_t, i_s, original_shape)
        save_result("examples/results", result, data_name, mode=1)
