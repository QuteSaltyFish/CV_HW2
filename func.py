import torch as t
import numpy as np
import cv2
from PIL import Image
import torchvision as tv
from torchvision import transforms
import json


class hw2():
    def __init__(self, dir, DEVICE=None):
        if DEVICE == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else 'cpu')
        else:
            self.DEVICE = t.device(DEVICE)
        self.img = Image.open(dir)
        self.img_gray = self.img.convert("L")
        self.img_tensor = transforms.ToTensor()(self.img).to(self.DEVICE)
        self.img_gray_tensor = transforms.ToTensor()(self.img_gray).to(self.DEVICE)
        self.load_config()

    def load_data(self, data):
        self.img_gray_tensor = t.tensor(
            data, dtype=t.float, device=self.DEVICE).unsqueeze(0)

    def load_config(self):
        file = open('config.json', "rb")
        self.config = json.load(file)

    def Erode(self, kernal, save_img=True, new_data=None):
        kernal = t.tensor(kernal, dtype=t.float, device=self.DEVICE)
        kernal_size = kernal.shape
        padding = [kernal_size[0]//2, kernal_size[0] //
                   2, kernal_size[1]//2, kernal_size[1]//2]
        if new_data is None:
            data = t.nn.ConstantPad2d(padding, 2)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, 2)(new_data)
        print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernal_size[0]+1,
                          data.shape[2]-kernal_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.min(
                        data[k, i:i+kernal_size[0], j:j+kernal_size[1]]-kernal)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Erode.gif')
            return out_img
        else:
            return output

    def Dilate(self, kernal, save_img=True, new_data=None):
        kernal = t.tensor(kernal, dtype=t.float, device=self.DEVICE)
        kernal_size = kernal.shape
        padding = [kernal_size[0] // 2, kernal_size[0] //
                   2, kernal_size[1] // 2, kernal_size[1] // 2]

        if new_data is None:
            data = t.nn.ConstantPad2d(padding, -1)(self.img_gray_tensor)
        else:
            data = t.nn.ConstantPad2d(padding, -1)(new_data)
        print(data)
        output = t.zeros([data.shape[0], data.shape[1]-kernal_size[0]+1,
                          data.shape[2]-kernal_size[1]+1], dtype=t.float).to(self.DEVICE)

        for k in range(output.shape[0]):
            for i in range(output.shape[1]):
                for j in range(output.shape[2]):
                    # print(data[k,i-1:i+1,j-1:j+1])
                    output[k, i, j] = t.max(
                        data[k, i:i+kernal_size[0], j:j+kernal_size[1]]+kernal)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Dilate.gif')
            return out_img
        else:
            return output

    def Opening(self, kernal, save_img=True, new_data=None):
        output = self.Erode(kernal, save_img=False, new_data=new_data)
        output = self.Dilate(kernal, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Opening.gif')
            return out_img
        else:
            return output

    def Closing(self, kernal, save_img=True, new_data=None):
        output = self.Dilate(kernal, save_img=False, new_data=new_data)
        output = self.Erode(kernal, save_img=False, new_data=output)
        if save_img:
            out_img = transforms.ToPILImage()(output.cpu())
            out_img.save('result/Closing.gif')
            return out_img
        else:
            return output

    def edge(self, kernel, save_img=True, new_data=None):
        pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)

        edge = pic_dilate - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection.gif')
            return out_img
        else:
            return edge

    def edge_ex(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        # pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)
        edge = pic_dilate - data

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection_external.gif')
            return out_img
        else:
            return edge

    def edge_in(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        pic_erode = self.Erode(kernel, False)
        # pic_dilate = self.Dilate(kernel, False)
        edge = data - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(edge.cpu())
            out_img.save('result/Edge_detection_internal.gif')
            return out_img
        else:
            return edge

    def grad(self, kernel, save_img=True, new_data=None):
        pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)

        grad = (pic_dilate - pic_erode)/2

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/Edge_detection.gif')
            return out_img
        else:
            return grad

    def grad_ex(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        # pic_erode = self.Erode(kernel, False)
        pic_dilate = self.Dilate(kernel, False)
        grad = pic_erode - data

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/Edge_detection_internal.gif')
            return out_img
        else:
            return grad

    def grad_in(self, kernel, save_img=True, new_data=None):
        if new_data is None:
            data = self.img_gray_tensor
        else:
            data = new_data
        pic_erode = self.Erode(kernel, False)
        # pic_dilate = self.Dilate(kernel, False)
        grad = data - pic_erode

        if save_img:
            out_img = transforms.ToPILImage()(grad.cpu())
            out_img.save('result/Edge_detection_internal.gif')
            return out_img
        else:
            return grad
