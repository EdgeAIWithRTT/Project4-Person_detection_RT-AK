#include <yolo_layer.h>
#include <math.h>

static int anchor[5][2]=
{
  {13, 24}, {33, 42}, {36, 87}, {94, 63}, {68, 118}
};
static int num_classes = 1;
static int input_dims[2] ={160,160};

int yolo_decode(float *out_data)
{
  int j=0,k=0,l=0;
  for(int i=0; i<5*5*5; i++)
  {
    float x_tmp = 1 / (1 + exp(-out_data[i*6+0]));
    float y_tmp = 1 / (1 + exp(-out_data[i*6+1]));
    float box_x = (x_tmp + k) / 5;
    float box_y = (y_tmp + l) / 5;
    
    float box_w = (exp(out_data[i*6+2])*anchor[j][0])/ input_dims[0];
    float box_h = (exp(out_data[i*6+3])*anchor[j][1])/ input_dims[1];
    
    float objectness = 1 / (1 + exp(-out_data[i*6+4]));
    
    float class_scores = 1 / (1 + exp(-out_data[i*6+5]));
   
//    printf("%d %d %d %f %f, %f %f, %f %f\n", j,k,l, box_x, box_y, box_w, box_h, objectness, class_scores);
    
    out_data[i*6+0] = box_x;
    out_data[i*6+1] = box_y;
    out_data[i*6+2] = box_w;
    out_data[i*6+3] = box_h;
    out_data[i*6+4] = objectness;
    out_data[i*6+5] = class_scores;
    
    if(j++>=4)
    {
      j = 0;
      if(k++>=4)
      {
        k = 0;
        if(l++>=4)
        {
          l = 0;
        }
      }
    }
  }
  return 0;
}
