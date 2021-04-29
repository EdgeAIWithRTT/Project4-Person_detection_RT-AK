# coding=utf-8
'''
@ Summary: 验证c代码的灰度图转化和缩放
@ Update:  

@ file:    val_c_gray_scaling.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/4/29 16:19
'''
import cv2
import numpy as np


def show_img(img, name="image"):
    assert img.size, "Error image datas"
    cv2.imshow(name, img)
    cv2.waitKey(2000)
    cv2.destroyWindow("results")


def img2gray(img_path):
    # 读取第一张图像
    img = cv2.imread(img_path)
    # 获取图像尺寸
    h, w = img.shape[0:2]
    # 自定义空白单通道图像，用于存放灰度图
    gray = np.zeros((h, w), dtype=img.dtype)
    # 对原图像进行遍历，然后分别对B\G\R按比例灰度化
    for i in range(h):
        for j in range(w):
            gray[i, j] = 0.11 * img[i, j, 0] + 0.59 * img[i, j, 1] + 0.3 * img[i, j, 2]  # Y=0.3R+0.59G+0.11B
    show_img(gray)
    return gray


def is_in_array(x, y, height, width):
    if x >= 0 and x < width and y >= 0 and y < height:
        return True
    else:
        return False


def bilinera_interpolation(in_array, height, width, out_height, out_width):
    h_times = out_height / height
    w_times = out_width / width
    x1, y1, x2, y2, f11, f12, f21, f22 = 0, 0, 0, 0, 0, 0, 0, 0
    x, y = 0., 0.
    out_array = np.empty([out_height, out_width], dtype=np.uint8)
    for i in range(out_height):
        for j in range(out_width):
            x = j / w_times
            y = i / h_times
            x1 = int(x - 1)
            x2 = int(x + 1)
            y1 = int(y + 1)
            y2 = int(y - 1)
            f11 = in_array[y1][x1] if is_in_array(x1, y1, height, width) else 0
            f12 = in_array[y2][x1] if is_in_array(x1, y2, height, width) else 0
            f21 = in_array[y1][x2] if is_in_array(x2, y1, height, width) else 0
            f22 = in_array[y2][x2] if is_in_array(x2, y2, height, width) else 0
            out_array[i][j] = int(((f11 * (x2 - x) * (y2 - y)) +
                                   (f21 * (x - x1) * (y2 - y)) +
                                   (f12 * (x2 - x) * (y - y1)) +
                                   (f22 * (x - x1) * (y - y1))) / ((x2 - x1) * (y2 - y1)))
    return out_array


if __name__ == "__main__":
    img_path = "imgs/person.jpg"
    # img_gray = img2gray(img_path)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_img(img_gray)
    (h, w) = img_gray.shape
    target_h, target_w = 240, 320
    img2 = bilinera_interpolation(img_gray, h, w, target_h, target_w)
    show_img(img2)

