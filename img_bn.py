# coding=utf-8
'''
@ Summary: 
@ Update:  

@ file:    img_bn.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/4/22 16:45
'''
import cv2
import numpy as np


def convert_img(img, shape=(320, 320), is_save=False):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, shape)
    result = np.zeros(gray.shape, dtype=np.float32)

    cv2.normalize(gray, result, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # tmp = gray / 255
    return result


def save_img_txt(iamge, filename="img_bn.txt"):
    img_list = iamge.flatten()
    img_list = map(lambda x:str(x), img_list)
    img_str = '{' + ", ".join(img_list) + '};'

    with open(filename, "w+") as f:
        f.write(img_str)

    print("Save image ok...")


if __name__ == "__main__":
    img_path = "person.jpg"
    result = convert_img(img_path)
    # _ = save_img_txt(result)
    image = cv2.imread(img_path)
    img_resize = cv2.resize(image, (320, 240))
    # cv2.imshow("image", img_resize)
    # cv2.waitKey(1000)
    cv2.imwrite("person_320_240.jpg", img_resize)

