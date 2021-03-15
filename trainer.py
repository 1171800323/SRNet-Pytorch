import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datagen import datagen_srnet, collate_fn
from loss import build_discriminator_loss, build_generator_loss
from model import Generator, Vgg19, Discriminator
from utils import *

device = torch.device(cfg.gpu)


def clip_grad(model):
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)


class Trainer:
    def __init__(self):
        self.data_loader = DataLoader(dataset=datagen_srnet(), batch_size=cfg.batch_size,
                                      shuffle=True, collate_fn=collate_fn, pin_memory=True,
                                      num_workers=16)

        self.vgg19 = Vgg19().to(device)

        self.G = Generator().to(device)
        self.D1 = Discriminator().to(device)
        self.D2 = Discriminator().to(device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), cfg.learning_rate, (cfg.beta1, cfg.beta2))
        self.d1_optimizer = torch.optim.Adam(self.D1.parameters(), cfg.learning_rate, (cfg.beta1, cfg.beta2))
        self.d2_optimizer = torch.optim.Adam(self.D2.parameters(), cfg.learning_rate, (cfg.beta1, cfg.beta2))

        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.d1_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d1_optimizer,
                                                                   (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.d2_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d2_optimizer,
                                                                   (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.g_writer, self.d_writer = None, None

    def train_step(self, data):
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = data
        i_t = i_t.to(device)
        i_s = i_s.to(device)
        t_sk = t_sk.to(device)
        t_t = t_t.to(device)
        t_b = t_b.to(device)
        t_f = t_f.to(device)
        mask_t = mask_t.to(device)

        inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]

        # ---------------
        # 训练鉴别器
        # 生成器输出的o_b和o_f，需要经过detach()截断梯度流，
        # 随着真实数据一起传入鉴别器网络，计算鉴别器损失，反向传播梯度，更新鉴别器参数
        o_sk, o_t, o_b, o_f = self.G(inputs)

        i_db_true = torch.cat([t_b, i_s], dim=1)
        i_db_pred = torch.cat([o_b.detach(), i_s], dim=1)

        i_df_true = torch.cat([t_f, i_t], dim=1)
        i_df_pred = torch.cat([o_f.detach(), i_t], dim=1)

        # 计算鉴别器损失
        o_db_true = self.D1(i_db_true)
        o_db_pred = self.D1(i_db_pred)

        o_df_true = self.D2(i_df_true)
        o_df_pred = self.D2(i_df_pred)

        db_loss = build_discriminator_loss(o_db_true, o_db_pred)
        df_loss = build_discriminator_loss(o_df_true, o_df_pred)
        d_loss_detail = [db_loss, df_loss]
        d_loss = torch.add(db_loss, df_loss)

        # 反向传播，更新梯度
        self.d1_optimizer.zero_grad()
        self.d2_optimizer.zero_grad()

        d_loss.backward()

        self.d1_optimizer.step()
        self.d2_optimizer.step()

        # 学习率衰减
        self.d1_scheduler.step()
        self.d2_scheduler.step()

        # 对鉴别器参数截断
        # clip_grad(self.D1)
        # clip_grad(self.D2)

        # ---------------
        # 训练生成器
        # 将未经过detach()的o_b和o_f输入到鉴别器网络，
        # 再计算生成器损失，反向传播梯度，更新生成器的参数
        i_db_pred = torch.cat([o_b, i_s], dim=1)
        i_df_pred = torch.cat([o_f, i_t], dim=1)

        o_db_pred = self.D1(i_db_pred)
        o_df_pred = self.D2(i_df_pred)

        # vgg损失
        i_vgg = torch.cat([t_f, o_f], dim=0)
        out_vgg = self.vgg19(i_vgg)

        out_g = [o_sk, o_t, o_b, o_f, mask_t]
        out_d = [o_db_pred, o_df_pred]

        g_loss, g_loss_detail = build_generator_loss(out_g, out_d, labels, out_vgg)

        # 反向传播，更新梯度
        self.g_optimizer.zero_grad()

        g_loss.backward()

        self.g_optimizer.step()

        # 学习率衰减
        self.g_scheduler.step()

        return d_loss, g_loss, d_loss_detail, g_loss_detail

    def train(self):
        if not cfg.train_name:
            train_name = get_train_name()
        else:
            train_name = cfg.train_name

        if sys.platform.startswith('win'):
            self.d_writer = SummaryWriter('model_logs\\train_logs\\' + train_name + '\\discriminator')
            self.g_writer = SummaryWriter('model_logs\\train_logs\\' + train_name + '\\generator')
        else:
            self.d_writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, train_name, 'discriminator'))
            self.g_writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, train_name, 'generator'))

        data_iter = iter(self.data_loader)

        for step in tqdm(range(cfg.max_iter)):
            global_step = step + 1

            # 数据读完后会抛出异常
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                data = next(data_iter)

            d_loss, g_loss, d_loss_detail, g_loss_detail = self.train_step(data)

            # 打印loss信息
            if global_step % cfg.show_loss_interval == 0 or step == 0:
                print_log("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))
                print('d_loss_detail: ' + str([float('%.4f' % d.data) for d in d_loss_detail]))
                print('g_loss_detail: ' + str([float('%.4f' % g.data) for g in g_loss_detail]))

            # 写tensorboard
            if global_step % cfg.write_log_interval == 0:
                self.write_summary(d_loss, d_loss_detail, g_loss, g_loss_detail, global_step)

            # 保存模型
            if global_step % cfg.save_ckpt_interval == 0:
                save_dir = os.path.join(cfg.checkpoint_save_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.save_checkpoint(save_dir)
                print_log("checkpoint saved in dir {}".format(save_dir), content_color=PrintColor['green'])

        print_log('training finished.', content_color=PrintColor['yellow'])

    def save_checkpoint(self, save_dir):
        os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pth'))
        torch.save({
            'D1': self.D1.state_dict(),
            'D2': self.D2.state_dict()
        }, os.path.join(save_dir, 'D.pth'))
        torch.save({
            'g_optimizer': self.g_optimizer.state_dict(),
            'd1_optimizer': self.d1_optimizer.state_dict(),
            'd2_optimizer': self.d2_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd1_scheduler': self.d1_scheduler.state_dict(),
            'd2_scheduler': self.d2_scheduler.state_dict()
        }, os.path.join(save_dir, 'optimizer-scheduler.pth'))

    def write_summary(self, d_loss, d_loss_detail, g_loss, g_loss_detail, step):
        self.d_writer.add_scalar('loss', d_loss, step)
        self.d_writer.add_scalar('l_db', d_loss_detail[0], step)
        self.d_writer.add_scalar('l_df', d_loss_detail[1], step)

        self.g_writer.add_scalar('loss', g_loss, step)
        self.g_writer.add_scalar('l_t_sk', g_loss_detail[0], step)
        self.g_writer.add_scalar('l_t_l1', g_loss_detail[1], step)
        self.g_writer.add_scalar('l_b_gan', g_loss_detail[2], step)
        self.g_writer.add_scalar('l_b_l1', g_loss_detail[3], step)
        self.g_writer.add_scalar('l_f_gan', g_loss_detail[4], step)
        self.g_writer.add_scalar('l_f_l1', g_loss_detail[5], step)
        self.g_writer.add_scalar('l_f_vgg_per', g_loss_detail[6], step)
        self.g_writer.add_scalar('l_f_vgg_style', g_loss_detail[7], step)


if __name__ == '__main__':
    Trainer().train()
