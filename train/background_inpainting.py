import os

import cv2
import numpy as np
import torch

import cfg
from datagen import background_inpainting_datagen, get_input_data
from loss import build_discriminator_loss, build_l_b_loss
from model import BackgroundInpaintingNet, SNDiscriminator
from utils import get_train_name, print_log, PrintColor, pre_process_img, save_result, get_log_writer

device = torch.device(cfg.gpu)


class BackgroundInpaintingTrainer:
    def __init__(self, data_dir):
        self.data_iter = background_inpainting_datagen(data_dir)

        self.G = BackgroundInpaintingNet().to(device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.learning_rate)
        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.D = SNDiscriminator().to(device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.2 * cfg.learning_rate)
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))

        self.writer = None

    def train(self):
        train_name = 'b_' + get_train_name()

        self.writer = get_log_writer(train_name)

        for step in range(cfg.max_iter):
            global_step = step + 1

            g_loss, g_loss_detail, d_loss = self.train_step(next(self.data_iter))

            # 打印loss信息
            if global_step % cfg.show_loss_interval == 0 or step == 0:
                print_log("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))

            # 写tensorboard
            if global_step % cfg.write_log_interval == 0:
                self.write_summary(g_loss, g_loss_detail, d_loss, global_step)

            # 生成example
            if global_step % cfg.gen_example_interval == 0:
                save_dir = os.path.join(cfg.example_result_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.predict_data_list(save_dir, get_input_data())
                print_log("example generated in dir {}".format(save_dir), content_color=PrintColor['green'])

            # 保存模型
            if global_step % cfg.save_ckpt_interval == 0:
                save_dir = os.path.join(cfg.checkpoint_save_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.save_checkpoint(save_dir)
                print_log("checkpoint saved in dir {}".format(save_dir), content_color=PrintColor['green'])

    def train_step(self, data):
        i_s, t_b = data

        i_s = i_s.to(device)
        t_b = t_b.to(device)

        o_b, _ = self.G(i_s)

        i_db_true = torch.cat([t_b, i_s], dim=1)
        i_db_pred = torch.cat([o_b, i_s], dim=1)
        i_db = torch.cat([i_db_true, i_db_pred], dim=0)

        o_db = self.D(i_db.detach())

        d_loss = build_discriminator_loss(o_db)

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        o_db = self.D(i_db)
        g_loss, g_loss_detail = build_l_b_loss(o_db, o_b, t_b)

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        self.d_scheduler.step()
        self.g_scheduler.step()

        return g_loss, g_loss_detail, d_loss

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def write_summary(self, g_loss, g_loss_detail, d_loss, step):
        self.writer.add_scalar('g_loss', g_loss, step)
        self.writer.add_scalar('l_b_gan', g_loss_detail[0], step)
        self.writer.add_scalar('l_b_l1', g_loss_detail[1], step)

        self.writer.add_scalar('d_loss', d_loss, step)

    def save_checkpoint(self, save_dir):
        os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'BackgroundInpainting.pth'))

    def predict(self, i_t, i_s, to_shape=None):
        assert i_t.shape == i_s.shape and i_t.dtype == i_s.dtype

        i_t, i_s, to_shape = pre_process_img(i_t, i_s, to_shape)
        o_b, _ = self.G(i_s.to(device))

        o_b = o_b.data.cpu()

        transpose_vector = [0, 2, 3, 1]
        o_b = o_b.permute(transpose_vector).numpy()

        o_b = cv2.resize(((o_b[0] + 1.) * 127.5).astype(np.uint8), to_shape)

        return o_b

    def predict_data_list(self, save_dir, input_data_list, mode=3):
        for data in input_data_list:
            i_t, i_s, original_shape, data_name = data
            result = self.predict(i_t, i_s, original_shape)
            save_result(save_dir, result, data_name, mode=mode)
