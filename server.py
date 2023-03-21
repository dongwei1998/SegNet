# coding=utf-8
# =============================================
# @Time      : 2022-08-11 9:42
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
from flask import Flask, jsonify, request
from utils import parameter
from utils.train import My_Model
import cv2,os
import numpy as np
from torchvision.utils import save_image
import torch


class Predictor(object):
    def __init__(self, args):
        self.args = args

        self.segnet_model = My_Model(self.args)




    def predict_(self,images):
        # 构建模型
        predict = self.segnet_model.predict_(images)

        image_name = len(os.listdir(args.predict_path))+1
        save_image(torch.cat([predict]), os.path.join(args.predict_path, f"predict_{image_name}.png"))

        return predict





Unlabelled = [0, 0, 0]
CellMembrane = [255, 255, 255]
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]

COLOR_DICT = np.array([
    Unlabelled, CellMembrane, Sky, Building, Pole, Road, Pavement,
    Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist
])

def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out

def saveResult(save_path, npyfile, num_class=2):
    for i, item in enumerate(npyfile):
        item = item[:, :, 0] if len(item.shape) == 3 else item  # 数据转换为二维的
        img = labelVisualize(num_class, COLOR_DICT, item)
        cv2.imwrite(save_path,img)
        # io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


if __name__ == '__main__':

    app = Flask(__name__)

    app.config['JSON_AS_ASCII'] = False

    model = 'server'
    args = parameter.parser_opt(model)

    my_model = Predictor(args)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            args.logger.info(f'=======Receipt of the request=======')
            data = request.files
            # data = request.get_json()
            if 'input' not in data:
                return 'input字段不存在', 500
            # 解码
            data_info = data['input'].read()
            img_array = np.frombuffer(data_info, np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_array = cv2.resize(img_array,(args.image_size,args.image_size))
            image = np.transpose(img_array,[2,0,1])
            # image = cv2.imread(data['input'], cv2.COLOR_RGB2GRAY)
            # image = tf.image.rgb_to_grayscale(img_array)
            image = torch.tensor([image],dtype=torch.float)
            # image /= 255.0  # 0 - 255 to 0.0 - 1.0
            args.logger.info(f'输入图片的形状为{image.shape}')
            if image is None:
                args.logger.error(f"not a image, please check your input")
                return jsonify(
                    {'code': 500,
                     'msg': 'not a image！！！'
                     })
            else:
                predictions = my_model.predict_(image)
                return jsonify({
                    'code': 200,
                    'msg': '成功',
                    'predictions': str(predictions),
                })
        except Exception as e:
            return jsonify(
                {'code': 500,
                 'msg': e
                 })

    # 启动
    app.run(host='0.0.0.0', port=3366)