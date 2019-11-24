import func
import torch as t
data = t.tensor([
    [209.0,  125,  191, 9, 168, 246, 158, 14],
    [232, 205, 101, 113, 42, 141,122,136],
    [33, 37, 168,98,31,36,91,200],
    [234,108,44,196,128,39,213,240],
    [162,235,181,204,246,66,150, 34],
    [25 ,203, 9,48,88,216,141,146],
    [72,246,71,126,150,66,235,121]
])/255
kernal = t.tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])/255.0
model = func.hw2('/home/wangmingke/Desktop/HomeWork/CV_HW2/src/img.jpg', 'cpu')
# model.load_data(data)
model.Erode(kernal)
model.Dilate(kernal)
model.edge_detection(kernal)
model.Opening(kernal)
model.Closing(kernal)