# device
gpu = 'cuda:0'

# pretrained vgg
vgg19_weights = 'model_logs/vgg19/vgg19-dcbb9e9d.pth'

# model parameters
lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
g_lr = 1e-4
d_lr = 1e-4
decay_rate = 0.9
decay_steps = 10000
beta1 = 0.9
beta2 = 0.999
max_iter = 500000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 50000
gen_example_interval = 1000

train_name = None
checkpoint_save_dir = 'model_logs/checkpoints'
tensorboard_dir = 'model_logs/train_logs'

# data
# 批处理大小
batch_size = 1
# 规定训练图片的形状
data_shape = [64, None]
# 训练数据地址
data_dir = '../datasets/srnet_data'
# 风格图像，包括背景和风格文字A
i_s_dir = 'i_s'
# 内容图像，灰色背景与标准字体的目标文字B
i_t_dir = 'i_t'
# 文字风格迁移到内容图像后生成的图片，灰色背景和风格化的目标文字B
t_t_dir = 't_t'
# 背景图像，擦除文字之后的图片
t_b_dir = 't_b'
# 最后结果，包括背景与风格化的目标文字B
t_f_dir = 't_f'
# 风格化的目标文字B的二进制掩码
mask_t_dir = 'mask_t'
# 骨架化风格文字B
t_sk_dir = 't_sk'

example_data_dir = 'examples/labels'
example_result_dir = 'examples/gen_logs'
