# coding=utf-8
'''
@ Summary: valid yolo-s model
@ Update:  

@ file:    inference_yolo-s.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/1/19 下午4:10

@ Update:  新增nms; 取消最大值选框
@ Date:    2021/1/26

@ Update:  新增 voc 测试集验证 h5 模型
@ Date:    2021/04/27
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
import logging

def get_label(label_file):
    """ 获取对应图片的标签，返回字典 """
    label_file = Path(label_file)
    assert label_file.exists(), FileNotFoundError(f"Not found {label_file}!!!")

    with label_file.open() as f:
        lines = f.readlines()

    # save {image: label}
    result = dict()
    for line in lines:
        img, label = line.strip().split()
        result[img] = label

    return result


def inference(img_path, model):
    img_raw = cv2.imread(str(img_path))
    # 灰度图 resize
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # shape: (160, 160)
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
    # RGB 图 resize
    # image_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(image_rgb, (320, 320),
    #                            interpolation=cv2.INTER_LINEAR)

    # normalize
    img = img / 255.0
    img = np.asarray(img).astype('float32')

    # expand channel
    img = np.expand_dims(img, axis=-1)

    # expand batch --> (1, 160, 160, 1)
    input = np.expand_dims(img, axis=0)

    # shape: (1, 5, 5, 30)
    yolo_output = model.predict(input)

    return img_raw, yolo_output


def draw_img(boxes, img):
    # show img
    x, y, w, h = boxes
    xmin = int(x - w / 2)
    xmax = int(x + w / 2)
    ymin = int(y - h / 2)
    ymax = int(y + h / 2)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # cv2.imwrite("./imgs/yolo-s_prediction.jpg", img)
    cv2.imshow('results', img)
    cv2.waitKey(1000)
    cv2.destroyWindow("results")


def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    num_anchors = len(anchors)  # anchor 的数量
    grid_size = prediction.shape[1:3]  # 将一张图片分割成5*5

    # shape: (125, 6)
    prediction = np.reshape(prediction,
                            (grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # generate x_y_offset grid map
    x_y_offset = [[[j, i]] * grid_size[0] for i in range(grid_size[0]) for j in range(grid_size[0])]
    x_y_offset = np.array(x_y_offset).reshape(grid_size[0] * grid_size[1] * num_anchors , 2)

    x_y_tmp = 1 / (1 + np.exp(-prediction[..., :2]))
    box_xy = (x_y_tmp + x_y_offset) / np.array(grid_size)[::-1]

    # Log space transform of the height and width
    anchors2 = np.array(anchors*(grid_size[0] * grid_size[1]))
    box_wh = (np.exp(prediction[..., 2:4]) * anchors2) / np.array(input_dims)[::-1]

    # sigmoid function
    objectness = 1 / (1 + np.exp(-prediction[..., 4:5]))

    # sigmoid function
    if use_softmax:
        class_scores = np.exp(prediction[..., 5:]) / np.sum(np.exp(prediction[..., 5:]))
    else:
        class_scores = 1 / (1 + np.exp(-prediction[..., 5:]))

    return np.concatenate((box_xy, box_wh), axis=-1), objectness, class_scores


def box_iou(boxes):
    """
    Calculate IoU value of 1st box with other boxes of a box array

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh

    Returns
    -------
    iou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h
    # left, top, right, bottom
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2

    # check IoU
    # np.maximun 诸位比较,取最大值
    # np.maximum([-2, -1, 0, 1, 2], 0)
    # array([0, 0, 0, 1, 2])
    inter_xmin = np.maximum(left[1:], left[0])
    inter_ymin = np.maximum(top[1:], top[0])
    inter_xmax = np.minimum(right[1:], right[0])
    inter_ymax = np.minimum(bottom[1:], bottom[0])

    inter_w = np.maximum(0., inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0., inter_ymax - inter_ymin + 1)
    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)
    return iou


def nms_boxes(boxes, classes, scores, iou_threshold=0.4, confidence=0.1,
              use_diou=True, is_soft=False, use_exp=False, sigma=0.5):
    import copy
    nboxes, nclasses, nscores = [], [], []

    # make a data copy to avoid breaking
    # during nms operation
    b_nms = copy.deepcopy(boxes)
    c_nms = copy.deepcopy(classes)
    s_nms = copy.deepcopy(scores)

    while len(s_nms) > 0:
        # pick the max box and store
        # make a data copy to avoid breaking
        i = np.argmax(s_nms)
        nboxes.append(copy.deepcopy(b_nms[i]))
        nclasses.append(copy.deepcopy(c_nms[i]))
        nscores.append(copy.deepcopy(s_nms[i]))

        # 把最大的scores 放到第一位
        b_nms[[i, 0], ...] = b_nms[[0, i], ...]
        s_nms[i], s_nms[0] = s_nms[0], s_nms[i]

        iou = box_iou(b_nms)

        b_nms = b_nms[1:]
        c_nms = c_nms[1:]
        s_nms = s_nms[1:]

        # hard nms
        keep_mast = np.where(iou <= iou_threshold)[0]

        # keep needed box for next loop
        b_nms = b_nms[keep_mast]
        c_nms = c_nms[keep_mast]
        s_nms = s_nms[keep_mast]

    # reformat result for output
    nboxes = np.array(nboxes)
    nclasses = np.array(nclasses)
    nscores = np.array(nscores)

    # nboxes[..., :2] -= (nboxes[..., 2:4] / 2)

    return nboxes, nclasses, nscores


def non_max_suppress(boxes, classes, scores, threshold=0.4):
    """ 简洁版 hard nms 实现, 单类别"""

    # center_xy, box_wh
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    # left, top, right, bottom
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2
    # sorted index, 从大到小
    order = np.argsort(scores)[::-1]
    ares = w * h

    keep = []  # 保留的有效的索引值, list
    # 保留置信度最高的box, 其余依次与其遍历, 删除大于阈值的box,剩
    # 下的继续保留最高置信度的box, 依次迭代
    while order.size > 0:
        keep.append(order[0])  # 永远保留置信度最高的索引
        # 最大置信度的左上角坐标分别与剩余所有的框的左上角坐标进行比较，分别保存较大值
        inter_xmin = np.maximum(left[order[0]], left[order[1:]])
        inter_ymin = np.maximum(top[order[0]], top[order[1:]])
        inter_xmax = np.minimum(right[order[0]], right[order[1:]])
        inter_ymax = np.minimum(bottom[order[0]], bottom[order[1:]])

        # 当前类所有框的面积
        # x1=3,x2=5,习惯上计算x方向长度就是x=3、4、5这三个像素，即5-3+1=3，
        # 而不是5-3=2，所以需要加1
        inter_w = np.maximum(0., inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0., inter_ymax - inter_ymin + 1)
        inter = inter_w * inter_h

        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        iou = inter / (ares[order[0]] + ares[order[1:]] - inter)

        # 计算iou的时候, 并没有计算第一个数, 所以索引对应的是order[1:]之后的, 所以需要加1
        indexs = np.where(iou <= threshold)[0]
        order = order[indexs+1]

    keep_boxes = boxes[keep]
    keep_classes = classes[keep]
    keep_scores = scores[keep]

    return keep_boxes, keep_classes, keep_scores


def filter(pred_xywh, objectness, class_scores, img_shape, iou_threshold=0.4, confidence=0.1):
    """ 得到真正的置信度，并且过滤 """
    # shape: (125, 1)
    box_scores = objectness * class_scores
    assert box_scores.shape[-1] == 1, "有不止一个类别, 该函数不可用, 仅对单类别使用"

    box_scores = np.squeeze(box_scores, axis=-1)
    # filter
    pos = np.where(box_scores >= confidence)
    if not pos:
        print("No person detected!!!")
        return
    # get all scores and boxes
    scores = box_scores[pos]
    boxes = pred_xywh[pos]
    classes = np.zeros(scores.shape, dtype=np.int8)  # 单类别

    # 相对坐标转为真实坐标
    boxes[..., :2] *= img_shape
    boxes[..., 2:] *= img_shape

    # nboxes2, nclasses2, nscores2 = nms_boxes(boxes, classes, scores, iou_threshold=iou_threshold)

    nboxes, nclasses, nscores = non_max_suppress(boxes, classes, scores, threshold=iou_threshold)

    return nboxes, nclasses, nscores


def inference_one_image(img_path, model, anchor):
    img, yolo_output = inference(img_path, model)
    img_shape = img.shape[:-1][::-1]  # weight, height
    pred_xywh, objectness, class_scores = yolo_decode(yolo_output, anchor, num_classes=1,
                                                      input_dims=(160, 160), scale_x_y=0,
                                                      use_softmax=False)
    boxes, classes, scores = filter(pred_xywh, objectness, class_scores, img_shape)
    return boxes, classes, scores, img

def main():
    logging.getLogger().setLevel(logging.INFO)
    img_path = "./imgs/person.jpg"
    model_path = './model/yolo-s.h5'
    root_path = Path("../test_voc_imgs")
    label_path = Path("/home/lebhoryi/Data/VOC/VOCdevkit/VOC2007/labels")
    label_file = Path("/home/lebhoryi/Data/VOC/VOCdevkit/VOC2007/ImageSets/Main/person_test.txt")


    # load model
    model = keras.models.load_model(model_path)

    # 预选框
    anchor = [[13, 24], [33, 42], [36, 87], [94, 63], [68, 118]]

    # 0 是检测单张图片， 1 是检测自己的数据集，并计算recall， 2 是计算模型的precision recall f1 scores
    inference_image = 0
    if inference_image == 0:
        boxes, classes, scores, img = inference_one_image(img_path, model, anchor)
        for box in boxes:
            draw_img(box, img)
        # boxes[..., :2] -= (boxes[..., 2:4] / 2)
    elif inference_image == 1:
        all_images = list(root_path.glob("*.jpg"))
        logging.info(f"all images numbers: {len(all_images)}")
        # dir
        count_tp, count_fn = 0, 0  # TP, FN
        for img_path in all_images:
            # shape: (1, 5, 5, 30)
            img, yolo_output = inference(img_path, model)
            img_shape = img.shape[:-1][::-1]   # weight, height

            pred_xywh, objectness, class_scores = yolo_decode(yolo_output, anchor, num_classes=1,
                                                              input_dims=(160, 160), scale_x_y=0,
                                                              use_softmax=False)

            # load grand truth boxes
            labels = img_path.parent / (img_path.stem + ".txt")
            with labels.open() as f:
                line = f.readline()
            line = line.split()[1:]
            gt_boxes = list(map(float, line))
            gt_boxes = np.array([gt_boxes])

            gt_boxes[:2] *= img_shape
            gt_boxes[2:] *= img_shape
            draw_img(gt_boxes, img)

            try:
                boxes, classes, scores = filter(pred_xywh, objectness, class_scores, img_shape,)
                boxes = np.concatenate((gt_boxes, boxes), axis=0)
                # draw_img(boxes[1], img)
                iou = box_iou(boxes)
                max_iou_index = np.argmax(scores)
                if iou[max_iou_index] > 0.6:
                    count_tp += 1
                else:
                    logging.info(f"wrong person detected image: {img_path.name}")
                    logging.info(f"iou: {iou[max_iou_index]}")
                    draw_img(boxes[max_iou_index+1], img)
                    count_fn += 1
            except:
                logging.info(f"no person detected image: {img_path.name}")
                count_fn += 1

        recall = count_tp / (count_tp+count_fn)
        recall = round(recall * 100, 2)
        print(f"recall: {recall} %")
        print(f"FRR: {round(100-recall, 2)} %")
    else:
        all_images = list(root_path.glob("*.jpg"))
        all_labels = get_label(label_file)
        count = 0
        tp, fn, fp, tn = 0, 0, 0, 0
        for img_path in all_images:
            if count % 200 == 0:
                print(f"Test progress: {count}/{len(all_images)}....")
            count += 1
            image_name = img_path.stem
            label = all_labels[image_name]
            boxes, classes, scores, img = inference_one_image(img_path, model, anchor)
            scores_filter = np.where(scores > 0.3)[0]
            if scores_filter.size == 0 and label == '-1':
                tn += 1
            elif scores_filter.size and label == '1':
                tp += 1
            elif scores_filter.size and label == '-1':
                fp += 1
            elif scores_filter.size == 0 and label == '1':
                fn += 1
        precision = round(tp / (tp + fp), 2)*100
        recall = round(tp / (tp + fn), 2)*100
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)*100
        f1_scores = round(2 * precision * recall / (precision + recall), 2)
        result = [f"precision: {str(precision)}% \n", f"recall: {str(recall)}% \n",
                 f"accuracy: {str(accuracy)}% \n", f"f1_scores: {str(f1_scores)}%\n"]
        with open("result.txt", "w+") as f:
             f.write("".join(result))
        for elem in result:
            print(elem)
        print("Done.")




if __name__ == "__main__":
    main()