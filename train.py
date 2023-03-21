# coding=utf-8
# =============================================
# @Time      : 2022-08-11 9:42
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
from utils import parameter
import os
from utils import dataloader,test_loader
from utils.train import My_Model
from utils.test import Tester



def main(args):
    args.logger.info(f'checkpoints dir {args.output_dir}')


    data_loader = dataloader.get_loader(args.train_data_dir, args.train_batch_size, args.image_size,
                             shuffle=True, num_workers=int(args.workers))

    segnet_model = My_Model(args)
    segnet_model.train(data_loader)


    segnet_model.test(data_loader)






if __name__ == '__main__':
    args = parameter.parser_opt(model='train')
    main(args)
    # print(args)
