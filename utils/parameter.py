# coding=utf-8
# =============================================
# @Time      : 2022-08-11 9:44
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.train_data_dir = os.path.join(os.environ.get('data_dir'),'train')
        args.test_data_dir = os.path.join(os.environ.get('data_dir'), 'test')
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        args.output_dir = os.environ.get('output_dir')
        args.logging_ini = os.environ.get('logging_ini')

        args.image_size = int(os.environ.get('image_size'))
        args.num_classes = int(os.environ.get('num_classes'))
        args.train_batch_size = int(os.environ.get('train_batch_size'))
        args.test_batch_size = int(os.environ.get('test_batch_size'))
        args.epoch = int(os.environ.get('epoch'))
        args.num_epoch = int(os.environ.get('num_epoch'))
        args.decay_epoch = int(os.environ.get('decay_epoch'))
        args.lr = int(os.environ.get('lr'))
        args.rho = float(os.environ.get('rho'))
        args.eps = float(os.environ.get('eps'))
        args.decay = float(os.environ.get('decay'))
        args.sample_step = int(os.environ.get('sample_step'))
        args.checkpoint_step = int(os.environ.get('checkpoint_step'))
        args.data_path = os.environ.get('data_path')

        args.sample_dir = os.environ.get('sample_dir')
        args.workers = int(os.environ.get('workers'))
        args.mode = os.environ.get('mode')
        args.num_test = int(os.environ.get('num_test'))
        args.transfer_learning = bool(0 if os.environ.get('transfer_learning')== 'False' else 1)
        args.gradient_loss_weight = float(os.environ.get('gradient_loss_weight'))

    elif model =='env':
        pass
    elif model == 'server':
        args.train_data_dir = os.path.join(os.environ.get('data_dir'), 'train')
        args.test_data_dir = os.path.join(os.environ.get('data_dir'), 'test')
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        args.output_dir = os.environ.get('output_dir')
        args.logging_ini = os.environ.get('logging_ini')

        args.image_size = int(os.environ.get('image_size'))
        args.num_classes = int(os.environ.get('num_classes'))
        args.train_batch_size = int(os.environ.get('train_batch_size'))
        args.test_batch_size = int(os.environ.get('test_batch_size'))
        args.epoch = int(os.environ.get('epoch'))
        args.num_epoch = int(os.environ.get('num_epoch'))
        args.decay_epoch = int(os.environ.get('decay_epoch'))
        args.lr = int(os.environ.get('lr'))
        args.rho = float(os.environ.get('rho'))
        args.eps = float(os.environ.get('eps'))
        args.decay = float(os.environ.get('decay'))
        args.sample_step = int(os.environ.get('sample_step'))
        args.checkpoint_step = int(os.environ.get('checkpoint_step'))
        args.data_path = os.environ.get('data_path')

        args.sample_dir = os.environ.get('sample_dir')
        args.workers = int(os.environ.get('workers'))
        args.mode = os.environ.get('mode')
        args.num_test = int(os.environ.get('num_test'))
        args.transfer_learning = bool(0 if os.environ.get('transfer_learning') == 'False' else 1)
        args.gradient_loss_weight = float(os.environ.get('gradient_loss_weight'))
        args.predict_path = os.environ.get('predict_path')
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')
    print(args)