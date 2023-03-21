from utils import transfer_model, model
from utils.loss import ImageGradientLoss,iou_loss
import os
from glob import glob
import torch
from utils.util import LambdaLR, AverageMeter
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.adadelta import Adadelta
import torch.nn as nn
from torchvision.utils import save_image

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class My_Model:
    def __init__(self, config):
        self.batch_size = config.train_batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = config.epoch
        self.num_epoch = config.num_epoch
        self.checkpoint_dir = config.output_dir
        self.model_path = config.output_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = config.num_classes
        self.eps = config.eps
        self.rho = config.rho
        self.decay = config.decay
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step
        self.sample_dir = config.sample_dir
        self.gradient_loss_weight = config.gradient_loss_weight
        self.decay_epoch = config.decay_epoch
        self.transfer_learning = config.transfer_learning

        self.build_model()
        self.optimizer = Adadelta(self.net.parameters(), lr=self.lr, eps=self.eps, rho=self.rho, weight_decay=self.decay)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.optimizer, self.decay_epoch)

    def build_model(self):
        if self.transfer_learning:
            self.net = transfer_model.MobileHairNet().to(self.device)
        else:
            self.net = model.MobileHairNet().to(self.device)
        self.load_model()



    def load_model(self):
        print("[*] Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print("[!] No checkpoint in ", str(self.model_path))
            return

        model_path = os.path.join(self.model_path, f"MobileHairNet_epoch-{self.epoch}.pth")
        model = glob(model_path)
        model.sort()
        if not model:
            print(f"[!] No Checkpoint in {model_path}")
            return

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(f"[*] Load Model from {model[-1]}: ")

    def train(self,dataloader):
        image_len = len(dataloader)
        bce_losses = AverageMeter()
        image_gradient_losses = AverageMeter()
        image_gradient_criterion = ImageGradientLoss().to(self.device)
        bce_criterion = nn.CrossEntropyLoss().to(self.device)
        num_step = 0
        for epoch in range(self.epoch, self.num_epoch):
            bce_losses.reset()
            image_gradient_losses.reset()
            for step, (image, gray_image, mask) in enumerate(dataloader):
                num_step += 1
                image = image.to(self.device)
                mask = mask.to(self.device)
                gray_image = gray_image.to(self.device)
                print(image.shape,type(image))
                pred = self.net(image)

                pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()

                # preds_flat.shape (N*224*224, 2)
                # masks_flat.shape (N*224*224, 1)
                image_gradient_loss = image_gradient_criterion(pred, gray_image)
                bce_loss = bce_criterion(pred_flat, mask_flat)

                loss = bce_loss + self.gradient_loss_weight * image_gradient_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bce_losses.update(bce_loss.item(), self.batch_size)
                image_gradient_losses.update(self.gradient_loss_weight * image_gradient_loss, self.batch_size)
                iou = iou_loss(pred, mask)

                # save sample images
                if num_step % 50 == 0:
                    print(f"Epoch: [{epoch}/{self.num_epoch}] | Step: [{num_step}/{image_len}] | "
                          f"Bce Loss: {bce_losses.avg:.4f} | Image Gradient Loss: {image_gradient_losses.avg:.4f} | "
                          f"IOU: {iou:.4f}")
                if num_step % self.sample_step == 0:
                    self.save_sample_imgs(image[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, num_step)
                    print('[*] Saved sample images')
            if num_step % self.checkpoint_step == 0:
                torch.save(self.net.state_dict(), f'{self.checkpoint_dir}/MobileHairNet_epoch-{num_step}.pth')
        torch.save(self.net.state_dict(), f'{self.checkpoint_dir}/MobileHairNet_epoch-{num_step}.pth')


    def test(self,data_loader):
        unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        for step, (image, gray_image, mask) in enumerate(data_loader):
            image = unnormal(image.to(self.device))
            mask = mask.to(self.device).repeat_interleave(3, 1)
            result = self.net(image)
            argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
            result = result[:, 1, :, :].unsqueeze(dim=1)
            result = result * argmax
            result = result.repeat_interleave(3, 1)
            torch.cat([image, result, mask])
            save_image(torch.cat([image, result, mask]), os.path.join(self.sample_dir, f"{step}.png"))
            print('[*] Saved sample images')



    def predict_(self,image):

        result = self.net(image.to(self.device))

        argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
        result = result[:, 1, :, :].unsqueeze(dim=1)
        result = result * argmax
        result = result.repeat_interleave(3, 1)


        return result

    def save_sample_imgs(self, real_img, real_mask, prediction, save_dir, epoch, step):
        data = [real_img, real_mask, prediction]
        names = ["Image", "Mask", "Prediction"]

        fig = plt.figure()
        for i, d in enumerate(data):
            d = d.squeeze()
            im = d.data.cpu().numpy()

            if i > 0:
                im = np.expand_dims(im, axis=0)
                im = np.concatenate((im, im, im), axis=0)

            im = (im.transpose(1, 2, 0) + 1) / 2

            f = fig.add_subplot(1, 3, i + 1)
            f.imshow(im)
            f.set_title(names[i])
            f.set_xticks([])
            f.set_yticks([])

        p = os.path.join(save_dir, "epoch-%s_step-%s.png" % (epoch, step))
        plt.savefig(p)