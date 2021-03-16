import torch

import cfg


def build_discriminator_loss(x_true, x_pred):
    d_loss = -torch.mean(torch.log(torch.clamp(x_true, cfg.epsilon, 1.0))
                         + torch.log(torch.clamp(1.0 - x_pred, cfg.epsilon, 1.0)))

    return d_loss


def build_dice_loss(x_t, x_o):
    intersection = torch.sum(x_t * x_o, dim=[1, 2, 3])
    union = torch.sum(x_t, dim=[1, 2, 3]) + torch.sum(x_o, dim=[1, 2, 3])

    return 1. - torch.mean((2 * intersection + cfg.epsilon) / (union + cfg.epsilon), dim=0)


def build_l1_loss_with_mask(x_t, x_o, mask):
    mask_ratio = 1. - torch.sum(mask) / torch.tensor(torch.numel(mask), dtype=torch.float32)
    l1 = torch.abs(x_t - x_o)

    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))


def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))


def build_gan_loss(x_pred):
    return -torch.mean(torch.log(torch.clamp(x_pred, cfg.epsilon, 1.0)))


def build_perceptual_loss(x):
    loss = []
    for f in x:
        loss.append(build_l1_loss(f[0], f[1]))
    loss = torch.stack(loss, dim=0)
    loss = torch.sum(loss)

    return loss


def build_gram_matrix(x):
    batch_size, channel, h, w = x.size()
    feature = x.view(batch_size, channel, h * w)
    feature_t = feature.transpose(1, 2)
    gram = torch.bmm(feature, feature_t) / torch.tensor(channel * h * w, dtype=torch.float32)

    return gram


def build_style_loss(x):
    loss = []
    for f in x:
        f_shape = torch.numel(f[0])
        f_norm = 1. / torch.tensor(f_shape, dtype=torch.float32)
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        loss.append(f_norm * build_l1_loss(gram_true, gram_pred))
    loss = torch.stack(loss, dim=0)
    loss = torch.sum(loss)

    return loss


def build_vgg_loss(x):
    split = []
    for f in x:
        split.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(split)
    l_style = build_style_loss(split)

    return l_per, l_style


def build_generator_loss(out_g, out_d, labels, out_vgg):
    o_sk, o_t, o_b, o_f, mask_t = out_g
    o_db_pred, o_df_pred = out_d
    o_vgg = out_vgg
    t_sk, t_t, t_b, t_f = labels

    l_t_sk = cfg.lt_alpha * build_dice_loss(t_sk, o_sk)
    l_t_l1 = build_l1_loss_with_mask(t_t, o_t, mask_t)
    l_t = l_t_l1 + l_t_sk

    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l1 = cfg.lb_beta * build_l1_loss(t_b, o_b)
    l_b = l_b_gan + l_b_l1

    l_f_gan = build_gan_loss(o_df_pred)
    l_f_l1 = cfg.lf_theta_1 * build_l1_loss(t_f, o_f)
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg)
    l_f_vgg_per = cfg.lf_theta_2 * l_f_vgg_per
    l_f_vgg_style = cfg.lf_theta_3 * l_f_vgg_style
    l_f = l_f_gan + l_f_l1 + l_f_vgg_per + l_f_vgg_style

    l = cfg.lt * l_t + cfg.lb * l_b + cfg.lf * l_f
    return l, [l_t_sk, l_t_l1, l_b_gan, l_b_l1, l_f_gan, l_f_l1, l_f_vgg_per, l_f_vgg_style]
