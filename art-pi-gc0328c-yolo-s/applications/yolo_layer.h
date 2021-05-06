#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_

typedef struct yolo_box{
    float x;
    float y;
    float w;
    float h;
    float objectness;
    float class_score;
} *yolo_box_t;

int yolo_decode(float *out_data);
#endif