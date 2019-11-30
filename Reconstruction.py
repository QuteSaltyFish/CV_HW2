import func
import torch as t

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from func import hw2
from PIL import Image, ImageTk
import json
config = json.load(open('config.json'))
DEVICE = config['DEVICE']
root = Tk()

# data = t.tensor([
#     [209.0,  125,  191, 9, 168, 246, 158, 14],
#     [232, 205, 101, 113, 42, 141,122,136],
#     [33, 37, 168,98,31,36,91,200],
#     [234,108,44,196,128,39,213,240],
#     [162,235,181,204,246,66,150, 34],
#     [25 ,203, 9,48,88,216,141,146],
#     [72,246,71,126,150,66,235,121]
# ])/255
# kernal = t.tensor([
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1]
# ])/255.0


Height = 3
Width = 3
H_flag = False
W_flag = False
Input_Flag = True
SE = None
i, j = 0, 0


def check_ready():
    global Height, Width, H_flag, W_flag, SE
    if H_flag and W_flag:
        l1.config(text='Please Input SE')
        l2.config(text='Please Input SE')
        l3.config(text='Please Input SE[0,0], range [0,255]')
        SE = t.zeros(Height, Width, device=DEVICE)


def get_H():
    try:
        global Height, H_flag, W_flag
        Height = int(e1.get())  # 获取e1的值，转为浮点数，如果不能转捕获异常
        l1.config(text='H='+str(Height))
        H_flag = True
        check_ready()
    except:
        messagebox.showwarning('警告', '请输入数字')


def get_W():
    try:
        global Width, W_flag, H_flag
        Width = int(e2.get())  # 获取e1的值，转为浮点数，如果不能转捕获异常
        l2.config(text='W='+str(Width))
        W_flag = True
        check_ready()
    except:
        messagebox.showwarning('警告', '请输入数字')


def get_SE():
    try:
        global SE, Height, Width, i, j, Input_Flag
        if Input_Flag:
            SE[i, j] = float(e3.get())
        if i == Height-1 and j == Width-1:
            l3.config(text='Please Choose the pic')
            Input_Flag = False
        else:

            i = (i+(j+1)//Width) % Height
            j = (j+1) % Width
            l3.config(
                text='Please Input SE[{},{}], range [0,255]'.format(i, j))

    except:
        messagebox.showwarning('警告', '请输入数字')


def printcoords():
    global H_flag, W_flag, SE, Input_Flag
    l1.config(text='PLS WAIT')
    l2.config(text='PLS WAIT')
    l3.config(text='PLS WAIT')
    # global kernal_size
    # global sigma
    try:
        if SE is None:
            raise NotImplementedError
        File = filedialog.askopenfilename(title='Choose an image.')
        pic = hw2(File, DEVICE)
        pic.MReconstruction(SE/255)

        # show the picture
        filename = ImageTk.PhotoImage(Image.open('result/Reconstruction.gif'))
        canvas.image = filename  # <--- keep reference of your image
        canvas.create_image(0, 0, anchor='nw', image=filename)

        # reset the program
        SE = None
        H_flag = False
        W_flag = False
        Input_Flag = True
        l1.config(text='Pls input Height')
        l2.config(text='Pls input Width')
    except NotImplementedError:
        messagebox.showwarning('警告', '请输入SE')


e1 = Entry(root)
e1.pack()
Button(root, text='click', command=get_H).pack()
l1 = Label(root, text='Pls input Height')
l1.pack()

e2 = Entry(root)
e2.pack()
Button(root, text='click', command=get_W).pack()
l2 = Label(root, text='Pls input Width')
l2.pack()

e3 = Entry(root)
e3.pack()
Button(root, text='click', command=get_SE).pack()
l3 = Label(root, text='Pls input H and W')
l3.pack()

# setting up a tkinter canvas with scrollbars
frame = Frame(root, bd=4, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set,
                yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH, expand=1)


# function to be called when mouse is clicked
Button(root, text='choose', command=printcoords).pack()

mainloop()
