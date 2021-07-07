#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_
#include <stdint.h>
typedef struct yolo_box{
    float x;
    float y;
    float w;
    float h;
    float objectness;
    float class_score;
} yolo_box, *yolo_box_t;

typedef struct sort_box{
    yolo_box_t box;
    int index;
    int cls;
} sort_box, *sort_box_t;

typedef struct
{
    float threshold;
    float nms_value;
    uint32_t anchor_number;
    uint32_t classes;
    uint32_t boxes_number;
    yolo_box *boxes;

} yolo_region_layer, *yolo_region_layer_t;

int yolo_decode(float *out_data);
void do_nms_sort(yolo_region_layer* region_layer, yolo_box *boxes);
int yolo_region_layer_init(yolo_region_layer *region_layer, int boxes_number, float nms_value, int classes, int anchor_number);
    
#endif