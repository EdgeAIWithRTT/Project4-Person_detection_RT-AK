/**
  ******************************************************************************
  * @file    person_yolo.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Apr 26 11:55:27 2021
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */


#include "person_yolo.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"



#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 5
#define AI_TOOLS_VERSION_MINOR 2
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_person_yolo
 
#undef AI_PERSON_YOLO_MODEL_SIGNATURE
#define AI_PERSON_YOLO_MODEL_SIGNATURE     "dd84ef15484776c44f584a7222f9dc01"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.2.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Apr 26 11:55:27 2021"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_PERSON_YOLO_N_BATCHES
#define AI_PERSON_YOLO_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2d_55_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv2d_54_scratch0_array;   /* Array #1 */
AI_STATIC ai_array conv2d_53_scratch0_array;   /* Array #2 */
AI_STATIC ai_array conv2d_51_scratch1_array;   /* Array #3 */
AI_STATIC ai_array conv2d_51_scratch0_array;   /* Array #4 */
AI_STATIC ai_array conv2d_48_scratch1_array;   /* Array #5 */
AI_STATIC ai_array conv2d_48_scratch0_array;   /* Array #6 */
AI_STATIC ai_array conv2d_43_scratch0_array;   /* Array #7 */
AI_STATIC ai_array conv2d_41_scratch1_array;   /* Array #8 */
AI_STATIC ai_array conv2d_41_scratch0_array;   /* Array #9 */
AI_STATIC ai_array conv2d_39_scratch1_array;   /* Array #10 */
AI_STATIC ai_array conv2d_39_scratch0_array;   /* Array #11 */
AI_STATIC ai_array conv2d_37_scratch0_array;   /* Array #12 */
AI_STATIC ai_array conv2d_35_scratch1_array;   /* Array #13 */
AI_STATIC ai_array conv2d_35_scratch0_array;   /* Array #14 */
AI_STATIC ai_array conv2d_33_scratch1_array;   /* Array #15 */
AI_STATIC ai_array conv2d_33_scratch0_array;   /* Array #16 */
AI_STATIC ai_array conv2d_31_scratch0_array;   /* Array #17 */
AI_STATIC ai_array conv2d_29_scratch1_array;   /* Array #18 */
AI_STATIC ai_array conv2d_29_scratch0_array;   /* Array #19 */
AI_STATIC ai_array conv2d_27_scratch1_array;   /* Array #20 */
AI_STATIC ai_array conv2d_27_scratch0_array;   /* Array #21 */
AI_STATIC ai_array conv2d_26_scratch0_array;   /* Array #22 */
AI_STATIC ai_array conv2d_24_scratch1_array;   /* Array #23 */
AI_STATIC ai_array conv2d_24_scratch0_array;   /* Array #24 */
AI_STATIC ai_array conv2d_21_scratch1_array;   /* Array #25 */
AI_STATIC ai_array conv2d_21_scratch0_array;   /* Array #26 */
AI_STATIC ai_array conv2d_17_scratch0_array;   /* Array #27 */
AI_STATIC ai_array conv2d_15_scratch1_array;   /* Array #28 */
AI_STATIC ai_array conv2d_15_scratch0_array;   /* Array #29 */
AI_STATIC ai_array conv2d_13_scratch1_array;   /* Array #30 */
AI_STATIC ai_array conv2d_13_scratch0_array;   /* Array #31 */
AI_STATIC ai_array conv2d_11_scratch1_array;   /* Array #32 */
AI_STATIC ai_array conv2d_11_scratch0_array;   /* Array #33 */
AI_STATIC ai_array conv2d_9_scratch0_array;   /* Array #34 */
AI_STATIC ai_array conv2d_7_scratch1_array;   /* Array #35 */
AI_STATIC ai_array conv2d_7_scratch0_array;   /* Array #36 */
AI_STATIC ai_array conv2d_4_scratch1_array;   /* Array #37 */
AI_STATIC ai_array conv2d_4_scratch0_array;   /* Array #38 */
AI_STATIC ai_array conv2d_2_scratch1_array;   /* Array #39 */
AI_STATIC ai_array conv2d_2_scratch0_array;   /* Array #40 */
AI_STATIC ai_array conv2d_55_bias_array;   /* Array #41 */
AI_STATIC ai_array conv2d_55_weights_array;   /* Array #42 */
AI_STATIC ai_array conv2d_54_bias_array;   /* Array #43 */
AI_STATIC ai_array conv2d_54_weights_array;   /* Array #44 */
AI_STATIC ai_array conv2d_53_bias_array;   /* Array #45 */
AI_STATIC ai_array conv2d_53_weights_array;   /* Array #46 */
AI_STATIC ai_array conv2d_51_bias_array;   /* Array #47 */
AI_STATIC ai_array conv2d_51_weights_array;   /* Array #48 */
AI_STATIC ai_array conv2d_48_bias_array;   /* Array #49 */
AI_STATIC ai_array conv2d_48_weights_array;   /* Array #50 */
AI_STATIC ai_array conv2d_43_bias_array;   /* Array #51 */
AI_STATIC ai_array conv2d_43_weights_array;   /* Array #52 */
AI_STATIC ai_array conv2d_41_bias_array;   /* Array #53 */
AI_STATIC ai_array conv2d_41_weights_array;   /* Array #54 */
AI_STATIC ai_array conv2d_39_bias_array;   /* Array #55 */
AI_STATIC ai_array conv2d_39_weights_array;   /* Array #56 */
AI_STATIC ai_array conv2d_37_bias_array;   /* Array #57 */
AI_STATIC ai_array conv2d_37_weights_array;   /* Array #58 */
AI_STATIC ai_array conv2d_35_bias_array;   /* Array #59 */
AI_STATIC ai_array conv2d_35_weights_array;   /* Array #60 */
AI_STATIC ai_array conv2d_33_bias_array;   /* Array #61 */
AI_STATIC ai_array conv2d_33_weights_array;   /* Array #62 */
AI_STATIC ai_array conv2d_31_bias_array;   /* Array #63 */
AI_STATIC ai_array conv2d_31_weights_array;   /* Array #64 */
AI_STATIC ai_array conv2d_29_bias_array;   /* Array #65 */
AI_STATIC ai_array conv2d_29_weights_array;   /* Array #66 */
AI_STATIC ai_array conv2d_27_bias_array;   /* Array #67 */
AI_STATIC ai_array conv2d_27_weights_array;   /* Array #68 */
AI_STATIC ai_array conv2d_26_bias_array;   /* Array #69 */
AI_STATIC ai_array conv2d_26_weights_array;   /* Array #70 */
AI_STATIC ai_array conv2d_24_bias_array;   /* Array #71 */
AI_STATIC ai_array conv2d_24_weights_array;   /* Array #72 */
AI_STATIC ai_array conv2d_21_bias_array;   /* Array #73 */
AI_STATIC ai_array conv2d_21_weights_array;   /* Array #74 */
AI_STATIC ai_array conv2d_17_bias_array;   /* Array #75 */
AI_STATIC ai_array conv2d_17_weights_array;   /* Array #76 */
AI_STATIC ai_array conv2d_15_bias_array;   /* Array #77 */
AI_STATIC ai_array conv2d_15_weights_array;   /* Array #78 */
AI_STATIC ai_array conv2d_13_bias_array;   /* Array #79 */
AI_STATIC ai_array conv2d_13_weights_array;   /* Array #80 */
AI_STATIC ai_array conv2d_11_bias_array;   /* Array #81 */
AI_STATIC ai_array conv2d_11_weights_array;   /* Array #82 */
AI_STATIC ai_array conv2d_9_bias_array;   /* Array #83 */
AI_STATIC ai_array conv2d_9_weights_array;   /* Array #84 */
AI_STATIC ai_array conv2d_7_bias_array;   /* Array #85 */
AI_STATIC ai_array conv2d_7_weights_array;   /* Array #86 */
AI_STATIC ai_array conv2d_4_bias_array;   /* Array #87 */
AI_STATIC ai_array conv2d_4_weights_array;   /* Array #88 */
AI_STATIC ai_array conv2d_2_bias_array;   /* Array #89 */
AI_STATIC ai_array conv2d_2_weights_array;   /* Array #90 */
AI_STATIC ai_array image_input_output_array;   /* Array #91 */
AI_STATIC ai_array conversion_0_output_array;   /* Array #92 */
AI_STATIC ai_array conv2d_2_output_array;   /* Array #93 */
AI_STATIC ai_array conv2d_4_output_array;   /* Array #94 */
AI_STATIC ai_array conv2d_7_output_array;   /* Array #95 */
AI_STATIC ai_array conv2d_9_output_array;   /* Array #96 */
AI_STATIC ai_array conv2d_11_output_array;   /* Array #97 */
AI_STATIC ai_array conv2d_13_output_array;   /* Array #98 */
AI_STATIC ai_array conv2d_15_output_array;   /* Array #99 */
AI_STATIC ai_array conv2d_17_output_array;   /* Array #100 */
AI_STATIC ai_array concat_20_output_array;   /* Array #101 */
AI_STATIC ai_array conv2d_21_output_array;   /* Array #102 */
AI_STATIC ai_array conv2d_24_output_array;   /* Array #103 */
AI_STATIC ai_array conv2d_26_output_array;   /* Array #104 */
AI_STATIC ai_array conv2d_27_output_array;   /* Array #105 */
AI_STATIC ai_array conv2d_29_output_array;   /* Array #106 */
AI_STATIC ai_array conv2d_31_output_array;   /* Array #107 */
AI_STATIC ai_array eltwise_32_output_array;   /* Array #108 */
AI_STATIC ai_array conv2d_33_output_array;   /* Array #109 */
AI_STATIC ai_array conv2d_35_output_array;   /* Array #110 */
AI_STATIC ai_array conv2d_37_output_array;   /* Array #111 */
AI_STATIC ai_array eltwise_38_output_array;   /* Array #112 */
AI_STATIC ai_array conv2d_39_output_array;   /* Array #113 */
AI_STATIC ai_array conv2d_41_output_array;   /* Array #114 */
AI_STATIC ai_array conv2d_43_output_array;   /* Array #115 */
AI_STATIC ai_array eltwise_44_output_array;   /* Array #116 */
AI_STATIC ai_array concat_47_output_array;   /* Array #117 */
AI_STATIC ai_array conv2d_48_output_array;   /* Array #118 */
AI_STATIC ai_array conv2d_51_output_array;   /* Array #119 */
AI_STATIC ai_array conv2d_53_output_array;   /* Array #120 */
AI_STATIC ai_array conv2d_54_output_array;   /* Array #121 */
AI_STATIC ai_array conv2d_55_output_array;   /* Array #122 */
AI_STATIC ai_array conversion_56_output_array;   /* Array #123 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2d_55_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv2d_54_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor conv2d_53_scratch0;   /* Tensor #2 */
AI_STATIC ai_tensor conv2d_51_scratch1;   /* Tensor #3 */
AI_STATIC ai_tensor conv2d_51_scratch0;   /* Tensor #4 */
AI_STATIC ai_tensor conv2d_48_scratch1;   /* Tensor #5 */
AI_STATIC ai_tensor conv2d_48_scratch0;   /* Tensor #6 */
AI_STATIC ai_tensor conv2d_43_scratch0;   /* Tensor #7 */
AI_STATIC ai_tensor conv2d_41_scratch1;   /* Tensor #8 */
AI_STATIC ai_tensor conv2d_41_scratch0;   /* Tensor #9 */
AI_STATIC ai_tensor conv2d_39_scratch1;   /* Tensor #10 */
AI_STATIC ai_tensor conv2d_39_scratch0;   /* Tensor #11 */
AI_STATIC ai_tensor conv2d_37_scratch0;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_35_scratch1;   /* Tensor #13 */
AI_STATIC ai_tensor conv2d_35_scratch0;   /* Tensor #14 */
AI_STATIC ai_tensor conv2d_33_scratch1;   /* Tensor #15 */
AI_STATIC ai_tensor conv2d_33_scratch0;   /* Tensor #16 */
AI_STATIC ai_tensor conv2d_31_scratch0;   /* Tensor #17 */
AI_STATIC ai_tensor conv2d_29_scratch1;   /* Tensor #18 */
AI_STATIC ai_tensor conv2d_29_scratch0;   /* Tensor #19 */
AI_STATIC ai_tensor conv2d_27_scratch1;   /* Tensor #20 */
AI_STATIC ai_tensor conv2d_27_scratch0;   /* Tensor #21 */
AI_STATIC ai_tensor conv2d_26_scratch0;   /* Tensor #22 */
AI_STATIC ai_tensor conv2d_24_scratch1;   /* Tensor #23 */
AI_STATIC ai_tensor conv2d_24_scratch0;   /* Tensor #24 */
AI_STATIC ai_tensor conv2d_21_scratch1;   /* Tensor #25 */
AI_STATIC ai_tensor conv2d_21_scratch0;   /* Tensor #26 */
AI_STATIC ai_tensor conv2d_17_scratch0;   /* Tensor #27 */
AI_STATIC ai_tensor conv2d_15_scratch1;   /* Tensor #28 */
AI_STATIC ai_tensor conv2d_15_scratch0;   /* Tensor #29 */
AI_STATIC ai_tensor conv2d_13_scratch1;   /* Tensor #30 */
AI_STATIC ai_tensor conv2d_13_scratch0;   /* Tensor #31 */
AI_STATIC ai_tensor conv2d_11_scratch1;   /* Tensor #32 */
AI_STATIC ai_tensor conv2d_11_scratch0;   /* Tensor #33 */
AI_STATIC ai_tensor conv2d_9_scratch0;   /* Tensor #34 */
AI_STATIC ai_tensor conv2d_7_scratch1;   /* Tensor #35 */
AI_STATIC ai_tensor conv2d_7_scratch0;   /* Tensor #36 */
AI_STATIC ai_tensor conv2d_4_scratch1;   /* Tensor #37 */
AI_STATIC ai_tensor conv2d_4_scratch0;   /* Tensor #38 */
AI_STATIC ai_tensor conv2d_2_scratch1;   /* Tensor #39 */
AI_STATIC ai_tensor conv2d_2_scratch0;   /* Tensor #40 */
AI_STATIC ai_tensor conv2d_55_bias;   /* Tensor #41 */
AI_STATIC ai_tensor conv2d_55_weights;   /* Tensor #42 */
AI_STATIC ai_tensor conv2d_54_bias;   /* Tensor #43 */
AI_STATIC ai_tensor conv2d_54_weights;   /* Tensor #44 */
AI_STATIC ai_tensor conv2d_53_bias;   /* Tensor #45 */
AI_STATIC ai_tensor conv2d_53_weights;   /* Tensor #46 */
AI_STATIC ai_tensor conv2d_51_bias;   /* Tensor #47 */
AI_STATIC ai_tensor conv2d_51_weights;   /* Tensor #48 */
AI_STATIC ai_tensor conv2d_48_bias;   /* Tensor #49 */
AI_STATIC ai_tensor conv2d_48_weights;   /* Tensor #50 */
AI_STATIC ai_tensor conv2d_43_bias;   /* Tensor #51 */
AI_STATIC ai_tensor conv2d_43_weights;   /* Tensor #52 */
AI_STATIC ai_tensor conv2d_41_bias;   /* Tensor #53 */
AI_STATIC ai_tensor conv2d_41_weights;   /* Tensor #54 */
AI_STATIC ai_tensor conv2d_39_bias;   /* Tensor #55 */
AI_STATIC ai_tensor conv2d_39_weights;   /* Tensor #56 */
AI_STATIC ai_tensor conv2d_37_bias;   /* Tensor #57 */
AI_STATIC ai_tensor conv2d_37_weights;   /* Tensor #58 */
AI_STATIC ai_tensor conv2d_35_bias;   /* Tensor #59 */
AI_STATIC ai_tensor conv2d_35_weights;   /* Tensor #60 */
AI_STATIC ai_tensor conv2d_33_bias;   /* Tensor #61 */
AI_STATIC ai_tensor conv2d_33_weights;   /* Tensor #62 */
AI_STATIC ai_tensor conv2d_31_bias;   /* Tensor #63 */
AI_STATIC ai_tensor conv2d_31_weights;   /* Tensor #64 */
AI_STATIC ai_tensor conv2d_29_bias;   /* Tensor #65 */
AI_STATIC ai_tensor conv2d_29_weights;   /* Tensor #66 */
AI_STATIC ai_tensor conv2d_27_bias;   /* Tensor #67 */
AI_STATIC ai_tensor conv2d_27_weights;   /* Tensor #68 */
AI_STATIC ai_tensor conv2d_26_bias;   /* Tensor #69 */
AI_STATIC ai_tensor conv2d_26_weights;   /* Tensor #70 */
AI_STATIC ai_tensor conv2d_24_bias;   /* Tensor #71 */
AI_STATIC ai_tensor conv2d_24_weights;   /* Tensor #72 */
AI_STATIC ai_tensor conv2d_21_bias;   /* Tensor #73 */
AI_STATIC ai_tensor conv2d_21_weights;   /* Tensor #74 */
AI_STATIC ai_tensor conv2d_17_bias;   /* Tensor #75 */
AI_STATIC ai_tensor conv2d_17_weights;   /* Tensor #76 */
AI_STATIC ai_tensor conv2d_15_bias;   /* Tensor #77 */
AI_STATIC ai_tensor conv2d_15_weights;   /* Tensor #78 */
AI_STATIC ai_tensor conv2d_13_bias;   /* Tensor #79 */
AI_STATIC ai_tensor conv2d_13_weights;   /* Tensor #80 */
AI_STATIC ai_tensor conv2d_11_bias;   /* Tensor #81 */
AI_STATIC ai_tensor conv2d_11_weights;   /* Tensor #82 */
AI_STATIC ai_tensor conv2d_9_bias;   /* Tensor #83 */
AI_STATIC ai_tensor conv2d_9_weights;   /* Tensor #84 */
AI_STATIC ai_tensor conv2d_7_bias;   /* Tensor #85 */
AI_STATIC ai_tensor conv2d_7_weights;   /* Tensor #86 */
AI_STATIC ai_tensor conv2d_4_bias;   /* Tensor #87 */
AI_STATIC ai_tensor conv2d_4_weights;   /* Tensor #88 */
AI_STATIC ai_tensor conv2d_2_bias;   /* Tensor #89 */
AI_STATIC ai_tensor conv2d_2_weights;   /* Tensor #90 */
AI_STATIC ai_tensor image_input_output;   /* Tensor #91 */
AI_STATIC ai_tensor conversion_0_output;   /* Tensor #92 */
AI_STATIC ai_tensor conv2d_2_output;   /* Tensor #93 */
AI_STATIC ai_tensor conv2d_4_output;   /* Tensor #94 */
AI_STATIC ai_tensor conv2d_7_output;   /* Tensor #95 */
AI_STATIC ai_tensor conv2d_9_output;   /* Tensor #96 */
AI_STATIC ai_tensor conv2d_11_output;   /* Tensor #97 */
AI_STATIC ai_tensor conv2d_13_output;   /* Tensor #98 */
AI_STATIC ai_tensor conv2d_15_output;   /* Tensor #99 */
AI_STATIC ai_tensor conv2d_17_output;   /* Tensor #100 */
AI_STATIC ai_tensor concat_20_output;   /* Tensor #101 */
AI_STATIC ai_tensor conv2d_21_output;   /* Tensor #102 */
AI_STATIC ai_tensor conv2d_24_output;   /* Tensor #103 */
AI_STATIC ai_tensor conv2d_26_output;   /* Tensor #104 */
AI_STATIC ai_tensor conv2d_27_output;   /* Tensor #105 */
AI_STATIC ai_tensor conv2d_29_output;   /* Tensor #106 */
AI_STATIC ai_tensor conv2d_31_output;   /* Tensor #107 */
AI_STATIC ai_tensor eltwise_32_output;   /* Tensor #108 */
AI_STATIC ai_tensor conv2d_33_output;   /* Tensor #109 */
AI_STATIC ai_tensor conv2d_35_output;   /* Tensor #110 */
AI_STATIC ai_tensor conv2d_37_output;   /* Tensor #111 */
AI_STATIC ai_tensor eltwise_38_output;   /* Tensor #112 */
AI_STATIC ai_tensor conv2d_39_output;   /* Tensor #113 */
AI_STATIC ai_tensor conv2d_41_output;   /* Tensor #114 */
AI_STATIC ai_tensor conv2d_43_output;   /* Tensor #115 */
AI_STATIC ai_tensor eltwise_44_output;   /* Tensor #116 */
AI_STATIC ai_tensor concat_47_output;   /* Tensor #117 */
AI_STATIC ai_tensor conv2d_48_output;   /* Tensor #118 */
AI_STATIC ai_tensor conv2d_51_output;   /* Tensor #119 */
AI_STATIC ai_tensor conv2d_53_output;   /* Tensor #120 */
AI_STATIC ai_tensor conv2d_54_output;   /* Tensor #121 */
AI_STATIC ai_tensor conv2d_55_output;   /* Tensor #122 */
AI_STATIC ai_tensor conversion_56_output;   /* Tensor #123 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conversion_0_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2d_2_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv2d_4_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain conv2d_7_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain conv2d_9_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain conv2d_11_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain conv2d_13_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain conv2d_15_chain;   /* Chain #7 */
AI_STATIC_CONST ai_tensor_chain conv2d_17_chain;   /* Chain #8 */
AI_STATIC_CONST ai_tensor_chain concat_20_chain;   /* Chain #9 */
AI_STATIC_CONST ai_tensor_chain conv2d_21_chain;   /* Chain #10 */
AI_STATIC_CONST ai_tensor_chain conv2d_24_chain;   /* Chain #11 */
AI_STATIC_CONST ai_tensor_chain conv2d_26_chain;   /* Chain #12 */
AI_STATIC_CONST ai_tensor_chain conv2d_27_chain;   /* Chain #13 */
AI_STATIC_CONST ai_tensor_chain conv2d_29_chain;   /* Chain #14 */
AI_STATIC_CONST ai_tensor_chain conv2d_31_chain;   /* Chain #15 */
AI_STATIC_CONST ai_tensor_chain eltwise_32_chain;   /* Chain #16 */
AI_STATIC_CONST ai_tensor_chain conv2d_33_chain;   /* Chain #17 */
AI_STATIC_CONST ai_tensor_chain conv2d_35_chain;   /* Chain #18 */
AI_STATIC_CONST ai_tensor_chain conv2d_37_chain;   /* Chain #19 */
AI_STATIC_CONST ai_tensor_chain eltwise_38_chain;   /* Chain #20 */
AI_STATIC_CONST ai_tensor_chain conv2d_39_chain;   /* Chain #21 */
AI_STATIC_CONST ai_tensor_chain conv2d_41_chain;   /* Chain #22 */
AI_STATIC_CONST ai_tensor_chain conv2d_43_chain;   /* Chain #23 */
AI_STATIC_CONST ai_tensor_chain eltwise_44_chain;   /* Chain #24 */
AI_STATIC_CONST ai_tensor_chain concat_47_chain;   /* Chain #25 */
AI_STATIC_CONST ai_tensor_chain conv2d_48_chain;   /* Chain #26 */
AI_STATIC_CONST ai_tensor_chain conv2d_51_chain;   /* Chain #27 */
AI_STATIC_CONST ai_tensor_chain conv2d_53_chain;   /* Chain #28 */
AI_STATIC_CONST ai_tensor_chain conv2d_54_chain;   /* Chain #29 */
AI_STATIC_CONST ai_tensor_chain conv2d_55_chain;   /* Chain #30 */
AI_STATIC_CONST ai_tensor_chain conversion_56_chain;   /* Chain #31 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_nl conversion_0_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d conv2d_2_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d conv2d_4_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d conv2d_7_layer; /* Layer #3 */
AI_STATIC ai_layer_conv2d conv2d_9_layer; /* Layer #4 */
AI_STATIC ai_layer_conv2d conv2d_11_layer; /* Layer #5 */
AI_STATIC ai_layer_conv2d conv2d_13_layer; /* Layer #6 */
AI_STATIC ai_layer_conv2d conv2d_15_layer; /* Layer #7 */
AI_STATIC ai_layer_conv2d conv2d_17_layer; /* Layer #8 */
AI_STATIC ai_layer_concat concat_20_layer; /* Layer #9 */
AI_STATIC ai_layer_conv2d conv2d_21_layer; /* Layer #10 */
AI_STATIC ai_layer_conv2d conv2d_24_layer; /* Layer #11 */
AI_STATIC ai_layer_conv2d conv2d_26_layer; /* Layer #12 */
AI_STATIC ai_layer_conv2d conv2d_27_layer; /* Layer #13 */
AI_STATIC ai_layer_conv2d conv2d_29_layer; /* Layer #14 */
AI_STATIC ai_layer_conv2d conv2d_31_layer; /* Layer #15 */
AI_STATIC ai_layer_eltwise eltwise_32_layer; /* Layer #16 */
AI_STATIC ai_layer_conv2d conv2d_33_layer; /* Layer #17 */
AI_STATIC ai_layer_conv2d conv2d_35_layer; /* Layer #18 */
AI_STATIC ai_layer_conv2d conv2d_37_layer; /* Layer #19 */
AI_STATIC ai_layer_eltwise eltwise_38_layer; /* Layer #20 */
AI_STATIC ai_layer_conv2d conv2d_39_layer; /* Layer #21 */
AI_STATIC ai_layer_conv2d conv2d_41_layer; /* Layer #22 */
AI_STATIC ai_layer_conv2d conv2d_43_layer; /* Layer #23 */
AI_STATIC ai_layer_eltwise eltwise_44_layer; /* Layer #24 */
AI_STATIC ai_layer_concat concat_47_layer; /* Layer #25 */
AI_STATIC ai_layer_conv2d conv2d_48_layer; /* Layer #26 */
AI_STATIC ai_layer_conv2d conv2d_51_layer; /* Layer #27 */
AI_STATIC ai_layer_conv2d conv2d_53_layer; /* Layer #28 */
AI_STATIC ai_layer_conv2d conv2d_54_layer; /* Layer #29 */
AI_STATIC ai_layer_conv2d conv2d_55_layer; /* Layer #30 */
AI_STATIC ai_layer_nl conversion_56_layer; /* Layer #31 */


/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1324, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2816, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_51_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3200, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_51_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4737, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1632, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_43_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 624, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3553, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1056, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 624, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3553, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1056, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_31_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 624, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3553, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1056, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 496, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2369, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 736, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 352, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 19200, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1777, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 19200, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3200, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 832, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 297, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 112, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 292, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 30, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7680, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_51_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_51_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11264, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_43_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_43_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 864, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 864, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_31_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_31_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 864, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 24, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 4, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 72, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  image_input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 25600, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 51200, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3200, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 19200, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 19200, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  concat_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 25600, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_41_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9600, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2400, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  concat_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8800, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3200, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1600, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 750, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  conversion_56_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 750, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_51_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07545003294944763f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_48_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043200161308050156f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_41_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11818278580904007f),
    AI_PACK_INTQ_ZP(55)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03861834481358528f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06505805253982544f),
    AI_PACK_INTQ_ZP(26)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04446561634540558f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_29_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08045352250337601f),
    AI_PACK_INTQ_ZP(42)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_27_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.055227190256118774f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_24_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11892437189817429f),
    AI_PACK_INTQ_ZP(24)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_21_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1147247701883316f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12874621152877808f),
    AI_PACK_INTQ_ZP(20)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12172151356935501f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08633572608232498f),
    AI_PACK_INTQ_ZP(-34)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1762116700410843f),
    AI_PACK_INTQ_ZP(-1)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.261122465133667f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_scratch1_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.20067691802978516f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_55_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 30,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.2471190934302285e-05f, 3.7683344999095425e-05f, 1.8523778635426424e-05f, 2.0392371880006976e-05f, 8.270990656455979e-05f, 7.161832036217675e-05f, 2.3783806682331488e-05f, 3.954373823944479e-05f, 1.7838712665252388e-05f, 2.396811578364577e-05f, 6.280287925619632e-05f, 7.219806866487488e-05f, 2.5881607143674046e-05f, 2.8552218282129616e-05f, 2.7806643629446626e-05f, 1.9101122234133072e-05f, 4.88780096929986e-05f, 6.71374291414395e-05f, 3.099302193732001e-05f, 2.7533444153959863e-05f, 2.1792660845676437e-05f, 1.8804306819220074e-05f, 5.551942012971267e-05f, 6.70769004500471e-05f, 3.026649937964976e-05f, 3.105466748820618e-05f, 2.302851135027595e-05f, 1.6896241504582576e-05f, 6.0795864555984735e-05f, 6.75130941090174e-05f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_55_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 30,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001002458157017827f, 0.001163369044661522f, 0.0005718704196624458f, 0.0006295580533333123f, 0.0025534394662827253f, 0.002211017534136772f, 0.0007342591998167336f, 0.0012208034750074148f, 0.0005507208989001811f, 0.0007399492315016687f, 0.001938865170814097f, 0.0022289156913757324f, 0.0007990230224095285f, 0.0008814707398414612f, 0.0008584531606175005f, 0.0005896943039260805f, 0.0015089733060449362f, 0.002072682371363044f, 0.000956823758315295f, 0.0008500189287588f, 0.0006727881263941526f, 0.0005805309629067779f, 0.001714008511044085f, 0.002070813672617078f, 0.0009343944257125258f, 0.0009587269159965217f, 0.000710941560100764f, 0.000521624693647027f, 0.0018769040470942855f, 0.0020842798985540867f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_54_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 256,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.6159023036598228e-05f, 3.941573959309608e-05f, 6.321472028503194e-05f, 4.194452412775718e-05f, 4.01763099944219e-05f, 4.48745813628193e-05f, 2.832863901858218e-05f, 2.4153998310794123e-05f, 4.6704983105883e-05f, 3.244991967221722e-05f, 3.7821933801751584e-05f, 3.0027680622879416e-05f, 3.472469325060956e-05f, 2.9245462428661995e-05f, 4.549390359898098e-05f, 5.075473745819181e-05f, 6.110293907113373e-05f, 6.841455615358427e-05f, 2.8572754672495648e-05f, 0.00010350869706599042f, 3.438907879171893e-05f, 4.136706775170751e-05f, 2.533574115659576e-05f, 2.903835047618486e-05f, 0.00010636950173648074f, 4.2688912799349055e-05f, 5.372956729843281e-05f, 2.639429112605285e-05f, 3.5570108593674377e-05f, 3.095432839472778e-05f, 3.192795702489093e-05f, 3.869248030241579e-05f, 3.2304345950251445e-05f, 2.9152215574868023e-05f, 5.62401364732068e-05f, 3.278291842434555e-05f, 3.247313725296408e-05f, 3.155191006953828e-05f, 2.7753396352636628e-05f, 2.2580496079172008e-05f, 2.8469756216509268e-05f, 2.6644418539945036e-05f, 3.5157274396624416e-05f, 3.143364665447734e-05f, 3.0730727303307503e-05f, 3.5248493077233434e-05f, 4.3330273911124095e-05f, 3.408692646189593e-05f, 3.4577409678604454e-05f, 2.6590827474137768e-05f, 4.5129869249649346e-05f, 3.651004226412624e-05f, 3.2345775252906606e-05f, 3.589924017433077e-05f, 6.22243678662926e-05f, 4.452235225471668e-05f, 2.4347156795556657e-05f, 3.454840771155432e-05f, 3.263626786065288e-05f, 4.797039946424775e-05f, 3.227158958907239e-05f, 5.855434210388921e-05f, 3.030698098882567e-05f, 2.5421935788472183e-05f, 3.663872121251188e-05f, 2.6870713554671966e-05f, 6.761027179891244e-05f, 2.7637426683213562e-05f, 3.5785342333838344e-05f, 4.0175127651309595e-05f, 3.906568235834129e-05f, 2.8737387765431777e-05f, 3.361854760441929e-05f, 4.768719009007327e-05f, 2.31925405387301e-05f, 2.5285040464950725e-05f, 5.234450873103924e-05f, 2.5272978746215813e-05f, 2.581283843028359e-05f, 5.0356647989246994e-05f, 7.345428457483649e-05f, 5.006001447327435e-05f, 2.7064188543590717e-05f, 4.140798409935087e-05f, 5.302976569510065e-05f, 5.1800387154798955e-05f, 4.34555477113463e-05f, 4.08505839004647e-05f, 5.3834115533391014e-05f, 6.649105489486828e-05f, 2.66493752860697e-05f, 3.304206256871112e-05f, 2.600171501399018e-05f, 3.817029210040346e-05f, 5.246191722108051e-05f, 4.780423114425503e-05f, 6.37589255347848e-05f, 8.465629070997238e-05f, 5.188911018194631e-05f, 3.4226781281176955e-05f, 4.593991252477281e-05f, 6.459442374762148e-05f, 3.790803748415783e-05f, 3.5389242839301005e-05f, 3.972447302658111e-05f, 4.7345627535833046e-05f, 5.519462138181552e-05f, 5.24556016898714e-05f, 6.129963730927557e-05f, 3.0134617190924473e-05f, 4.706215258920565e-05f, 3.075506174354814e-05f, 2.7667911126627587e-05f, 4.2169289372395724e-05f, 3.065375130972825e-05f, 5.371314182411879e-05f, 2.433071676932741e-05f, 2.583346031315159e-05f, 3.067286525038071e-05f, 5.108700497657992e-05f, 2.8463158741942607e-05f, 5.21896145073697e-05f, 3.332781489007175e-05f, 0.00010501355427550152f, 4.4679873099084944e-05f, 5.723471622331999e-05f, 3.9887807361083105e-05f, 3.255942283431068e-05f, 4.7061774239409715e-05f, 4.698499105870724e-05f, 9.570191468810663e-05f, 2.3985676307347603e-05f, 5.4466396250063553e-05f, 2.493816464266274e-05f, 3.2970649044727907e-05f, 6.040286461939104e-05f, 4.812115730601363e-05f, 0.0001604703429620713f, 4.174126297584735e-05f, 2.578428939159494e-05f, 4.453842484508641e-05f, 2.3024929760140367e-05f, 3.3672116842353716e-05f, 3.4581673389766365e-05f, 3.6462428397499025e-05f, 5.144682290847413e-05f, 4.1184979636454955e-05f, 4.647908281185664e-05f, 2.6845767933991738e-05f, 5.3693576774094254e-05f, 4.441961573320441e-05f, 5.761813372373581e-05f, 5.06551077705808e-05f, 3.395514067960903e-05f, 5.133317245054059e-05f, 4.6616401959909126e-05f, 3.540268517099321e-05f, 6.592859426746145e-05f, 3.368180841789581e-05f, 2.719248914218042e-05f, 3.0083592719165608e-05f, 3.8393794966395944e-05f, 2.6476132916286588e-05f, 2.7297346605337225e-05f, 3.4039981983369216e-05f, 4.15321956097614e-05f, 7.596848445245996e-05f, 3.1806001061340794e-05f, 5.9999743825756013e-05f, 3.893040775437839e-05f, 3.0308567147585563e-05f, 3.2003881642594934e-05f, 3.330377512611449e-05f, 4.284434544388205e-05f, 4.3263851694064215e-05f, 6.010604920447804e-05f, 5.295906885294244e-05f, 2.3874952603364363e-05f, 3.364908116054721e-05f, 3.2638061384204775e-05f, 5.739510015700944e-05f, 4.703125523519702e-05f, 4.121160236536525e-05f, 3.8712587411282584e-05f, 4.57042915513739e-05f, 4.497576446738094e-05f, 4.7241079300874844e-05f, 2.1903706510784104e-05f, 5.3251267672749236e-05f, 5.018726733396761e-05f, 2.5750096028787084e-05f, 3.466116322670132e-05f, 4.0311711927643046e-05f, 6.34827883914113e-05f, 5.203317050472833e-05f, 5.2602841606130823e-05f, 2.638635123730637e-05f, 4.378761877887882e-05f, 3.451362135820091e-05f, 3.8513553590746596e-05f, 3.319144161650911e-05f, 3.832394941127859e-05f, 2.8762467991327867e-05f, 4.5097658585291356e-05f, 3.161749555147253e-05f, 3.4371609217487276e-05f, 3.858954005409032e-05f, 5.079081893200055e-05f, 5.442750989459455e-05f, 3.368940087966621e-05f, 3.573111825971864e-05f, 3.7203750252956524e-05f, 6.539426976814866e-05f, 3.946239667129703e-05f, 3.289969026809558e-05f, 2.9352964702411555e-05f, 2.9772611014777794e-05f, 4.56350899185054e-05f, 3.3844131394289434e-05f, 3.62172577297315e-05f, 3.1734762160340324e-05f, 3.517559525789693e-05f, 2.7756344934459776e-05f, 4.789900776813738e-05f, 4.856043233303353e-05f, 5.124727977090515e-05f, 3.772406853386201e-05f, 4.276713661965914e-05f, 5.493749267770909e-05f, 2.3872915335232392e-05f, 3.253043178119697e-05f, 5.6739507272141054e-05f, 3.2671374356141314e-05f, 2.5594163162168115e-05f, 4.2140105506405234e-05f, 2.4483661036356352e-05f, 2.3032094759400934e-05f, 3.486227069515735e-05f, 2.81238117167959e-05f, 2.9493945476133376e-05f, 4.6885746996849775e-05f, 6.164668593555689e-05f, 3.388662662473507e-05f, 5.3171050240052864e-05f, 4.748599531012587e-05f, 3.0312570743262768e-05f, 3.625140743679367e-05f, 4.855951920035295e-05f, 3.0725710530532524e-05f, 5.0288523198105395e-05f, 6.116589793236926e-05f, 3.951387043343857e-05f, 2.799136927933432e-05f, 9.184335067402571e-05f, 3.655817999970168e-05f, 5.753960431320593e-05f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_54_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 256,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005747062969021499f, 0.0008659525774419308f, 0.001388809410855174f, 0.0009215092868544161f, 0.000882662134245038f, 0.0009858817793428898f, 0.000622372142970562f, 0.0005306564853526652f, 0.0010260952403768897f, 0.0007129154982976615f, 0.0008309371187351644f, 0.0006596996099688113f, 0.0007628916064277291f, 0.000642514496576041f, 0.0009994881693273783f, 0.0011150671634823084f, 0.001342414179816842f, 0.0015030483482405543f, 0.000627735280431807f, 0.002274056663736701f, 0.0007555183256044984f, 0.0009088227525353432f, 0.0005566190229728818f, 0.0006379642873071134f, 0.0023369076661765575f, 0.000937863253057003f, 0.0011804233072325587f, 0.0005798750789836049f, 0.0007814651471562684f, 0.0006800577975809574f, 0.0007014481234364212f, 0.0008500627591274679f, 0.000709717336576432f, 0.0006404658779501915f, 0.001235579838976264f, 0.0007202313863672316f, 0.0007134255720302463f, 0.0006931864772923291f, 0.0006097342120483518f, 0.0004960870719514787f, 0.000625472457613796f, 0.0005853702896274626f, 0.0007723953458480537f, 0.0006905883201397955f, 0.0006751453038305044f, 0.0007743993774056435f, 0.0009519538143649697f, 0.0007488800911232829f, 0.0007596559007652104f, 0.0005841929232701659f, 0.0009914904367178679f, 0.0008021152461878955f, 0.000710627471562475f, 0.0007886961102485657f, 0.0013670516200363636f, 0.0009781434200704098f, 0.0005349001148715615f, 0.0007590187015011907f, 0.0007170095341280103f, 0.0010538960341364145f, 0.0007089976570568979f, 0.0012864222517237067f, 0.0006658357451669872f, 0.0005585126928053796f, 0.0008049422758631408f, 0.0005903419223614037f, 0.0014853784814476967f, 0.0006071864045225084f, 0.0007861937629058957f, 0.0008826361736282706f, 0.0008582619484513998f, 0.000631352246273309f, 0.0007385899079963565f, 0.0010476739844307303f, 0.000509533507283777f, 0.0005555051611736417f, 0.0011499939719215035f, 0.0005552401416935027f, 0.0005671007093042135f, 0.0011063212295994163f, 0.001613769680261612f, 0.0010998042998835444f, 0.0005945925367996097f, 0.0009097216534428298f, 0.0011650489177554846f, 0.001138039748184383f, 0.0009547060471959412f, 0.0008974756929092109f, 0.0011827201815322042f, 0.001460789586417377f, 0.0005854791961610317f, 0.0007259247358888388f, 0.0005712502752430737f, 0.0008385904366150498f, 0.001152573386207223f, 0.0010502453660592437f, 0.001400765497237444f, 0.0018598746974021196f, 0.0011399890063330531f, 0.0007519526407122612f, 0.0010092868469655514f, 0.0014191210502758622f, 0.0008328288095071912f, 0.0007774916011840105f, 0.0008727353997528553f, 0.001040169969201088f, 0.0012126101646572351f, 0.0011524346191436052f, 0.0013467356329783797f, 0.0006620489875786006f, 0.0010339420987293124f, 0.0006756799411959946f, 0.0006078561418689787f, 0.0009264472755603492f, 0.0006734541966579854f, 0.0011800624197348952f, 0.0005345389363355935f, 0.000567553797736764f, 0.0006738741649314761f, 0.0011223669862374663f, 0.0006253275205381215f, 0.0011465910356491804f, 0.000732202606741339f, 0.0023071179166436195f, 0.000981604098342359f, 0.0012574304128065705f, 0.0008763237856328487f, 0.0007153212791308761f, 0.0010339338332414627f, 0.0010322468588128686f, 0.002102543832734227f, 0.0005269584944471717f, 0.0011966112069785595f, 0.0005478843813762069f, 0.0007243557483889163f, 0.001327033736743033f, 0.0010572081664577127f, 0.003525487845763564f, 0.0009170436533167958f, 0.0005664735217578709f, 0.0009784965077415109f, 0.0005058511742390692f, 0.0007397668086923659f, 0.0007597494986839592f, 0.0008010691963136196f, 0.0011302720522508025f, 0.0009048223146237433f, 0.0010211322223767638f, 0.0005897938972339034f, 0.0011796326143667102f, 0.0009758863598108292f, 0.0012658539926633239f, 0.0011128783226013184f, 0.0007459847838617861f, 0.0011277751764282584f, 0.0010241491254419088f, 0.0007777869468554854f, 0.0014484324492514133f, 0.0007399797905236483f, 0.0005974112427793443f, 0.0006609279662370682f, 0.000843500776682049f, 0.0005816731136292219f, 0.0005997149273753166f, 0.0007478487095795572f, 0.000912450545001775f, 0.0016690059565007687f, 0.0006987688248045743f, 0.001318177324719727f, 0.00085528998170048f, 0.0006658706115558743f, 0.0007031161803752184f, 0.0007316744886338711f, 0.0009412780636921525f, 0.0009504944900982082f, 0.001320512848906219f, 0.0011634957045316696f, 0.000524525938089937f, 0.0007392607512883842f, 0.0007170489407144487f, 0.0012609540717676282f, 0.0010332632809877396f, 0.0009054071852006018f, 0.0008505045552738011f, 0.001004110323265195f, 0.000988104729913175f, 0.0010378730949014425f, 0.00048121821600943804f, 0.0011699151946231723f, 0.0011026000138372183f, 0.0005657222936861217f, 0.000761495903134346f, 0.0008856368367560208f, 0.0013946988619863987f, 0.001143153989687562f, 0.0011556694516912103f, 0.0005797006306238472f, 0.0009620015043765306f, 0.0007582544349133968f, 0.0008461318211629987f, 0.000729206542018801f, 0.0008419663063250482f, 0.0006319032399915159f, 0.000990782747976482f, 0.000694627407938242f, 0.000755134504288435f, 0.0008478012168779969f, 0.001115859835408628f, 0.0011957569513469934f, 0.0007401465554721653f, 0.0007850024849176407f, 0.0008173558162525296f, 0.0014366935938596725f, 0.0008669776725582778f, 0.0007227968308143318f, 0.0006448762724176049f, 0.000654095783829689f, 0.0010025899391621351f, 0.0007435459410771728f, 0.000795682892203331f, 0.0006972037372179329f, 0.0007727978518232703f, 0.0006097989971749485f, 0.0010523275705054402f, 0.0010668588802218437f, 0.0011258882004767656f, 0.0008287870441563427f, 0.0009395818342454731f, 0.0012069611111655831f, 0.0005244811763986945f, 0.0007146843126975f, 0.0012465508189052343f, 0.0007177808438427746f, 0.0005622964818030596f, 0.0009258061181753874f, 0.0005378990899771452f, 0.0005060085677541792f, 0.0007659142138436437f, 0.0006178721669130027f, 0.0006479736184701324f, 0.001030066516250372f, 0.001354360138066113f, 0.0007444795919582248f, 0.0011681528994813561f, 0.0010432538110762835f, 0.0006659585633315146f, 0.0007964331307448447f, 0.0010668388567864895f, 0.0006750351167283952f, 0.0011048245942220092f, 0.0013437974266707897f, 0.0008681084727868438f, 0.0006149624241515994f, 0.002017772290855646f, 0.0008031728211790323f, 0.0012641287175938487f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_53_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.3530304790474474e-05f, 0.00022641890973318368f, 0.0002453993365634233f, 0.00015507831994909793f, 7.844563515391201e-05f, 0.00013178416702430695f, 0.00019563722889870405f, 5.7622441090643406e-05f, 5.967093602521345e-05f, 6.19915226707235e-05f, 8.030837489059195e-05f, 3.760926119866781e-05f, 0.0001354971172986552f, 0.0002493918873369694f, 0.00010272918007103726f, 0.00017898855730891228f, 9.696475899545476e-05f, 0.000149172919918783f, 4.63640499219764e-05f, 0.00016354580293409526f, 0.00012475108087528497f, 8.10151977930218e-05f, 0.00010951863077934831f, 0.00013144160038791597f, 0.00011818233178928494f, 0.00023236661218106747f, 0.00013319966092240065f, 4.5875920477556065e-05f, 0.00010437126911710948f, 0.00017125424346886575f, 8.578338020015508e-05f, 9.233666787622496e-05f, 0.00018134599667973816f, 0.00015050578804221004f, 0.00011955954687437043f, 7.117688801372424e-05f, 0.00010552919411566108f, 4.2831867176573724e-05f, 6.894732359796762e-05f, 5.544336454477161e-05f, 9.432887600269169e-05f, 0.00013234683137852699f, 0.00023457327915821224f, 0.00010190852481173351f, 0.0002605656336527318f, 9.132312698056921e-05f, 4.265023017069325e-05f, 4.845476723858155e-05f, 5.141014844411984e-05f, 4.051607902511023e-05f, 0.00017000484513118863f, 0.0002071381313726306f, 0.00018500330043025315f, 0.00012870982754975557f, 0.00017747000674717128f, 9.456438419874758e-05f, 0.00016686695744283497f, 6.578669126611203e-05f, 4.8774618335301057e-05f, 9.835937817115337e-05f, 5.2473773394012824e-05f, 5.046265141572803e-05f, 0.00017648751963861287f, 0.00011188734788447618f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_53_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008439404773525894f, 0.005698847584426403f, 0.0061765750870108604f, 0.0039032415952533484f, 0.0019744362216442823f, 0.003316939575597644f, 0.004924088250845671f, 0.0014503272250294685f, 0.0015018867561593652f, 0.0015602947678416967f, 0.0020213203970342875f, 0.0009466057526879013f, 0.0034103926736861467f, 0.006277065258473158f, 0.0025856406427919865f, 0.004505049902945757f, 0.0024405531585216522f, 0.003754605771973729f, 0.0011669592931866646f, 0.004116363823413849f, 0.0031399205327033997f, 0.002039110753685236f, 0.0027565276250243187f, 0.0033083173912018538f, 0.0029745884239673615f, 0.005848547909408808f, 0.0033525668550282717f, 0.0011546732857823372f, 0.002626971108838916f, 0.004310381133109331f, 0.0021591235417872667f, 0.0023240663576871157f, 0.004564385395497084f, 0.003788153175264597f, 0.0030092522501945496f, 0.001791485701687634f, 0.0026561154518276453f, 0.0010780560551211238f, 0.0017353686271235347f, 0.001395480940118432f, 0.0023742092307657003f, 0.0033311014994978905f, 0.005904088728129864f, 0.0025649850722402334f, 0.006558303255587816f, 0.002298556035384536f, 0.0010734843090176582f, 0.0012195815797895193f, 0.0012939670123159885f, 0.0010197688825428486f, 0.004278934560716152f, 0.0052135600708425045f, 0.004656437784433365f, 0.0032395601738244295f, 0.004466828890144825f, 0.002380136866122484f, 0.004199955612421036f, 0.0016558171482756734f, 0.0012276320485398173f, 0.0024756549391895533f, 0.0013207378797233105f, 0.0012701189843937755f, 0.0044420999474823475f, 0.0028161469381302595f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_51_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00011720712791429833f, 0.0001251678477274254f, 7.500511856051162e-05f, 5.704732029698789e-05f, 0.00026803815853782f, 9.91431443253532e-05f, 0.0001188536625704728f, 0.00010179635137319565f, 0.0002105252060573548f, 0.0002047980815405026f, 0.00020039014634676278f, 0.00011374983296263963f, 0.000190329083125107f, 0.00010361622116761282f, 8.449361484963447e-05f, 0.00015252690354827791f, 0.0001466735702706501f, 0.0001944799441844225f, 8.463627455057576e-05f, 7.562324753962457e-05f, 6.45056861685589e-05f, 0.00017194794781971723f, 6.969244714127854e-05f, 0.00011722780618583784f, 0.00010728293273132294f, 9.820738341659307e-05f, 0.00012878244160674512f, 7.58010137360543e-05f, 6.360150291584432e-05f, 0.00014211717643775046f, 0.00010846841178135946f, 0.00018374741193838418f, 0.00011710072431014851f, 0.00013301341095939279f, 0.0001695975661277771f, 0.00016015232540667057f, 3.9959882997209206e-05f, 6.468368519563228e-05f, 0.0001913849264383316f, 9.134966239798814e-05f, 0.00010802978067658842f, 0.00010146210115635768f, 0.00011191717203473672f, 7.255227683344856e-05f, 0.0002460276009514928f, 0.00010718800331233069f, 0.00012871914077550173f, 0.00016995545593090355f, 0.00011055143113480881f, 0.0001337415596935898f, 0.00011097483366029337f, 0.0001183948406833224f, 9.018442506203428e-05f, 0.00021082439343445003f, 0.0001750377705320716f, 9.072889224626124e-05f, 6.854740786366165e-05f, 0.00013383713667280972f, 5.5863802117528394e-05f, 0.0001461073843529448f, 6.319016392808408e-05f, 0.00018188320973422378f, 9.819074330152944e-05f, 7.768559589749202e-05f, 0.00010400677274446934f, 0.00018001775606535375f, 0.0001433344150427729f, 8.467831503367051e-05f, 6.807632598793134e-05f, 8.998312841868028e-05f, 0.0001377446169499308f, 0.00027143783518113196f, 0.00019048061221837997f, 6.111485708970577e-05f, 0.00012725021224468946f, 0.00011542686843313277f, 6.238928472157568e-05f, 0.0001564130070619285f, 6.539661262650043e-05f, 6.725000275764614e-05f, 8.848604193190113e-05f, 8.537507528671995e-05f, 7.893392466939986e-05f, 0.00010391103569418192f, 8.417763456236571e-05f, 0.00014674034900963306f, 8.63341338117607e-05f, 0.00012682225496973842f, 9.930197120411322e-05f, 0.0001730403455439955f, 8.563385927118361e-05f, 9.621383651392534e-05f, 0.00011196043487871066f, 9.132058039540425e-05f, 8.605970651842654e-05f, 6.937798752915114e-05f, 0.0001328387443209067f, 6.504767952719703e-05f, 8.40730790514499e-05f, 0.00010547848069109023f, 9.557205339660868e-05f, 0.00011323839135002345f, 0.00010566813580226153f, 6.856671825516969e-05f, 0.00017276153084822f, 0.0001646182790864259f, 0.0001247444743057713f, 0.00014716642908751965f, 0.0001309422659687698f, 0.00011286881635896862f, 0.00043459460721351206f, 0.00011843619722640142f, 0.00012063218309776857f, 9.88343235803768e-05f, 0.00020040158415213227f, 0.00010056736937258393f, 0.00010969680442940444f, 0.0003529801615513861f, 6.792233762098476e-05f, 0.0001050929568009451f, 0.00011205668852198869f, 5.634748958982527e-05f, 0.00010367781214881688f, 0.00015514963888563216f, 0.00010582830873318017f, 8.438567601842806e-05f, 8.294601138914004e-05f, 0.00011864067346323282f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_51_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004719643387943506f, 0.00504020182415843f, 0.003020272124558687f, 0.0022971555590629578f, 0.010793238878250122f, 0.003992251120507717f, 0.004785945173352957f, 0.004099089186638594f, 0.008477333001792431f, 0.008246716111898422f, 0.008069219067692757f, 0.004580426495522261f, 0.007664084900170565f, 0.0041723707690835f, 0.0034023504704236984f, 0.0061418842524290085f, 0.005906184669584036f, 0.007831229828298092f, 0.0034080951008945704f, 0.0030451626516878605f, 0.0025974856689572334f, 0.006923921871930361f, 0.002806343836709857f, 0.0047204759903252125f, 0.004320020321756601f, 0.003954570274800062f, 0.0051857526414096355f, 0.003052320796996355f, 0.0025610763113945723f, 0.005722709931433201f, 0.00436775665730238f, 0.007399057038128376f, 0.004715358838438988f, 0.005356123670935631f, 0.006829277612268925f, 0.0064489408396184444f, 0.0016090864082798362f, 0.002604653127491474f, 0.007706601172685623f, 0.0036784266121685505f, 0.004350094124674797f, 0.004085629712790251f, 0.004506629891693592f, 0.0029215021058917046f, 0.00990692712366581f, 0.004316197708249092f, 0.005183203611522913f, 0.006843688897788525f, 0.004451634828001261f, 0.00538544449955225f, 0.004468684084713459f, 0.004767469596117735f, 0.0036315054167062044f, 0.008489380590617657f, 0.007048341445624828f, 0.0036534296814352274f, 0.002760235918685794f, 0.005389292724430561f, 0.002249498153105378f, 0.005883385427296162f, 0.002544512739405036f, 0.007323990110307932f, 0.003953900188207626f, 0.0031282082200050354f, 0.004188097547739744f, 0.0072488728910684586f, 0.005771724972873926f, 0.003409787779673934f, 0.002741266507655382f, 0.0036233996506780386f, 0.00554663734510541f, 0.010930134914815426f, 0.007670186925679445f, 0.0024609453976154327f, 0.005124053917825222f, 0.004647956695407629f, 0.0025122633669525385f, 0.006298367865383625f, 0.0026333611458539963f, 0.002707992447540164f, 0.0035631158389151096f, 0.0034378448035568f, 0.0031784754246473312f, 0.004184242337942123f, 0.0033896267414093018f, 0.005908873863518238f, 0.0034764637239277363f, 0.005106820724904537f, 0.003998646512627602f, 0.006967910099774599f, 0.0034482653718441725f, 0.003874294925481081f, 0.00450837193056941f, 0.003677255706861615f, 0.0034654131159186363f, 0.0027936813421547413f, 0.005349089857190847f, 0.002619310515001416f, 0.003385416464880109f, 0.004247359465807676f, 0.0038484518881887197f, 0.004559832159429789f, 0.004254996310919523f, 0.002761013340204954f, 0.0069566830061376095f, 0.006628774106502533f, 0.0050231534987688065f, 0.005926030687987804f, 0.005272723734378815f, 0.0045449500903487206f, 0.017500057816505432f, 0.004769134800881147f, 0.004857562016695738f, 0.003979815635830164f, 0.008069680072367191f, 0.004049601033329964f, 0.004417221061885357f, 0.01421364489942789f, 0.0027350657619535923f, 0.004231835249811411f, 0.004512247629463673f, 0.0022689751349389553f, 0.004174850881099701f, 0.006247494835406542f, 0.004261446185410023f, 0.003398003987967968f, 0.003340032184496522f, 0.004777368623763323f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_48_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00011815596371889114f, 0.00010955899051623419f, 9.443534509046003e-05f, 0.00015558359154965729f, 0.00014990626368671656f, 0.00010089363058796152f, 7.691306382184848e-05f, 0.00017387286061421037f, 7.717134576523677e-05f, 0.00011933536734431982f, 0.0001707087067188695f, 9.702658280730247e-05f, 0.00010181868128711358f, 0.00016938874614425004f, 0.00011427821300458163f, 0.00012369935575406998f, 0.00015727612480986863f, 9.545726061332971e-05f, 8.223386976169422e-05f, 8.55298712849617e-05f, 8.552736107958481e-05f, 0.0001460516796214506f, 0.00011915041250176728f, 8.571297803428024e-05f, 0.0001055544416885823f, 0.00012225290993228555f, 9.292751929024234e-05f, 0.00014976499369367957f, 0.00011921456462005153f, 0.00011186553456354886f, 0.00012307983706705272f, 0.00013691822823602706f, 0.00011744202492991462f, 0.00011112229549326003f, 8.11222053016536e-05f, 0.00012372755736578256f, 7.951082079671323e-05f, 0.00018998967425432056f, 0.0001011759668472223f, 0.00016522069927304983f, 0.00011101069685537368f, 0.00012278786743991077f, 0.0001307157799601555f, 0.00013057398609817028f, 0.0001357903383905068f, 0.00011688857193803415f, 0.00010702496365411207f, 0.0002103934093611315f, 0.00011813556193374097f, 0.00015382775745820254f, 0.00010836673754965886f, 0.00013893353752791882f, 0.00010172918700845912f, 0.00012500870798248798f, 0.00010737025149865076f, 0.00010588087752694264f, 0.0001144619527622126f, 0.0001011909480439499f, 0.00013208822929300368f, 0.00011239986633881927f, 9.60161050898023e-05f, 0.00013401216710917652f, 0.0001173837881651707f, 0.00013240570842754096f, 7.327454659389332e-05f, 0.00011789322888944298f, 5.2186616812832654e-05f, 0.00012183933722553775f, 8.839762449497357e-05f, 0.00011749139957828447f, 0.0001110364610212855f, 7.864065264584497e-05f, 0.00010853599087568f, 0.0001450015843147412f, 0.0001413947029504925f, 0.00014700926840305328f, 0.0001294936373597011f, 8.342113142134622e-05f, 0.00010366675269324332f, 0.00019416272698435932f, 0.0001266193576157093f, 8.608052303316072e-05f, 0.0001141286120400764f, 0.0001129344163928181f, 0.00010470263077877462f, 0.00017221129382960498f, 0.000138629533466883f, 9.73221831372939e-05f, 0.00010267203469993547f, 6.0124708397779614e-05f, 0.0001114023762056604f, 0.00010698613186832517f, 0.00011707463272614405f, 0.00012839790724683553f, 0.00013769365614280105f, 9.70691253314726e-05f, 0.00013598566874861717f, 0.0002062392159132287f, 0.00012490515655372292f, 0.00012245359539519995f, 0.00010866780939977616f, 0.0001416705927113071f, 0.00010226338781649247f, 0.0001445272791897878f, 0.00013501843204721808f, 0.0001442290813429281f, 0.00011314585572108626f, 0.00010969341383315623f, 0.00011355568130966276f, 8.565058669773862e-05f, 8.406070264754817e-05f, 0.00011889390589203686f, 0.0001256923278560862f, 9.950809180736542e-05f, 9.486260387348011e-05f, 0.00011864382395287976f, 0.00013834383571520448f, 8.607519703218713e-05f, 0.0001338804286206141f, 9.637806215323508e-05f, 0.000144155666930601f, 0.00012601584603544325f, 0.00011851071758428589f, 7.728952914476395e-05f, 0.0001074091560440138f, 0.00010256854875478894f, 6.130077235866338e-05f, 0.00010513729648664594f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_48_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013943205121904612f, 0.0012928703799843788f, 0.0011144011514261365f, 0.0018359919777140021f, 0.001768995658494532f, 0.0011906132567673922f, 0.0009076262940652668f, 0.0020518177188932896f, 0.0009106742218136787f, 0.0014082383131608367f, 0.002014478435739875f, 0.0011449794983491302f, 0.0012015295214951038f, 0.001998902065679431f, 0.001348560443148017f, 0.0014597362605854869f, 0.0018559650052338839f, 0.001126460381783545f, 0.0009704154217615724f, 0.0010093104792758822f, 0.0010092809097841382f, 0.0017235089326277375f, 0.0014060556422919035f, 0.001011471264064312f, 0.001245614024810493f, 0.001442667213268578f, 0.0010966077679768205f, 0.0017673284746706486f, 0.001406812691129744f, 0.0013200892135500908f, 0.0014524254947900772f, 0.0016157279023900628f, 0.0013858955353498459f, 0.0013113184832036495f, 0.0009572970448061824f, 0.00146006909199059f, 0.000938281649723649f, 0.0022420070599764585f, 0.001193945063278079f, 0.0019497163593769073f, 0.0013100014766678214f, 0.001448980183340609f, 0.001542534795589745f, 0.0015408615581691265f, 0.0016024181386455894f, 0.0013793644029647112f, 0.0012629671255126595f, 0.0024827849119901657f, 0.001394079765304923f, 0.001815271913073957f, 0.0012788010062649846f, 0.0016395100392401218f, 0.0012004734016954899f, 0.0014751874841749668f, 0.001267041778191924f, 0.0012494662078097463f, 0.0013507286785170436f, 0.0011941217817366123f, 0.00155873061157763f, 0.001326394616626203f, 0.001133055193349719f, 0.001581434509716928f, 0.0013852083357051015f, 0.0015624770894646645f, 0.0008646893547847867f, 0.0013912200229242444f, 0.0006158374599181116f, 0.0014377868501469493f, 0.0010431519476696849f, 0.0013864781940355897f, 0.0013103055534884334f, 0.0009280130616389215f, 0.0012807983439415693f, 0.0017111171036958694f, 0.0016685533337295055f, 0.0017348090186715126f, 0.001528112799860537f, 0.0009844258893281221f, 0.001223337952978909f, 0.0022912519052624702f, 0.0014941943809390068f, 0.001015808549709618f, 0.0013467950047925115f, 0.0013327026972547174f, 0.0012355620274320245f, 0.0020322101190686226f, 0.001635922584682703f, 0.0011484677670523524f, 0.0012115996796637774f, 0.0007095123291946948f, 0.001314623630605638f, 0.0012625089148059487f, 0.001381560112349689f, 0.0015151824336498976f, 0.0016248784959316254f, 0.001145481481216848f, 0.0016047230456024408f, 0.002433762652799487f, 0.0014739655889570713f, 0.0014450354501605034f, 0.0012823538854718208f, 0.0016718090046197176f, 0.0012067772913724184f, 0.0017055199714377522f, 0.001593309105373919f, 0.0017020009690895677f, 0.001335197826847434f, 0.0012944566551595926f, 0.0013400340685620904f, 0.0010107350535690784f, 0.000991973327472806f, 0.0014030287275090814f, 0.001483254716731608f, 0.0011742629576474428f, 0.001119443099014461f, 0.0014000775991007686f, 0.0016325510805472732f, 0.0010157456854358315f, 0.0015798797830939293f, 0.0011373264715075493f, 0.0017011346062645316f, 0.0014870724407956004f, 0.0013985068071633577f, 0.0009120688773691654f, 0.0012675009202212095f, 0.0012103784829378128f, 0.0007233907235786319f, 0.001240691402927041f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_43_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00020710482203867286f, 0.00023548172612208873f, 0.0002985734899993986f, 0.00025764547172002494f, 0.0002026356232818216f, 0.0002011705655604601f, 0.00016354821855202317f, 0.0001119885710068047f, 0.00023250709637068212f, 0.00021169846877455711f, 0.00020223356841597706f, 0.00019543802773114294f, 0.00031780911376699805f, 0.00013878167374059558f, 0.00010473589645698667f, 0.00024031491193454713f, 0.0001992077159229666f, 9.085794590646401e-05f, 0.0001982400572160259f, 0.00018310542509425431f, 0.0001396675652358681f, 0.0001357581786578521f, 0.0001440166524844244f, 0.00021010565978940576f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_43_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004964113235473633f, 0.0056442818604409695f, 0.007156533654779196f, 0.006175526883453131f, 0.004856990650296211f, 0.004821874666959047f, 0.003920101560652256f, 0.0026842637453228235f, 0.005572982598096132f, 0.005074218846857548f, 0.004847353789955378f, 0.004684471059590578f, 0.007617594208568335f, 0.0033264700323343277f, 0.0025104237720370293f, 0.005760129075497389f, 0.004774827044457197f, 0.002177781891077757f, 0.004751632921397686f, 0.0043888697400689125f, 0.003347703954204917f, 0.003253999399021268f, 0.0034519475884735584f, 0.005036040674895048f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_41_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(3.8424641388701275e-05f, 0.00018119471496902406f, 0.0001816112344386056f, 0.0001546328712720424f, 0.0001787400833563879f, 0.00010453116556163877f, 0.00015387962048407644f, 0.00022990450088400394f, 0.00010481882782187313f, 0.00014423408720176667f, 0.00016578572103753686f, 0.00023460097145289183f, 0.00013938252232037485f, 0.00011053628259105608f, 0.0002024366840487346f, 7.717211701674387e-05f, 0.0001786531211109832f, 0.0001366896613035351f, 0.00013316948025021702f, 0.00016390728706028312f, 0.00011773347068810835f, 0.0001537516654934734f, 0.00017477382789365947f, 0.00019150246225763112f, 0.00013738554844167084f, 0.00015083540347404778f, 0.0001764582411851734f, 0.0001330821542069316f, 0.00018282834207639098f, 0.0001162226326414384f, 0.00017344375373795629f, 0.00010690259659895673f, 0.00013085766113363206f, 0.00017743161879479885f, 0.00015575803990941495f, 0.0001302986202063039f, 9.817253885557875e-05f, 0.00015156685549300164f, 0.0002172153617721051f, 7.152275793487206e-05f, 9.837732068262994e-05f, 0.00012199012417113408f, 0.00010714894597185776f, 0.00017674785340204835f, 0.00015458308917004615f, 0.00014784578524995595f, 0.00018426275346428156f, 0.00021072941308375448f, 0.00011860027007060125f, 0.00020896420755889267f, 9.955793939298019e-05f, 0.0001760203012963757f, 0.00018965567869599909f, 0.0002538589760661125f, 8.968239853857085e-05f, 0.00013768463395535946f, 9.710679296404123e-05f, 0.00014737393939867616f, 0.00022934237495064735f, 0.00026849276036955416f, 0.00012236287875566632f, 0.00019260983390267938f, 0.00013567146379500628f, 0.0003370283229742199f, 0.00011170742800459266f, 0.0001925222168210894f, 0.00014773165457881987f, 0.0002764361852314323f, 0.00014306019875220954f, 0.0001070900761988014f, 0.0002000276290345937f, 0.00015746901044622064f, 0.00014800183998886496f, 9.98572213575244e-05f, 0.0001337544381385669f, 9.843980660662055e-05f, 0.00012664860696531832f, 0.0005487861344590783f, 0.00014466843276750296f, 0.00016133542521856725f, 0.00014438650396186858f, 0.00012829432671424001f, 0.00020559190306812525f, 0.00023136936943046749f, 0.000232184975175187f, 0.00012348535528872162f, 0.00017796237079892308f, 0.00013138516806066036f, 0.00034088120446540415f, 9.960155875887722e-05f, 0.0001676170650171116f, 0.0001528386346763f, 0.00010718896373873577f, 0.00012578953464981169f, 0.00016026341472752392f, 0.00011268061643932015f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_41_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017585691530257463f, 0.008292685262858868f, 0.008311747573316097f, 0.0070770373567938805f, 0.008180344477295876f, 0.00478404713794589f, 0.007042563054710627f, 0.010521971620619297f, 0.004797212313860655f, 0.006601118948310614f, 0.007587466388940811f, 0.010736913420259953f, 0.006379079073667526f, 0.005058881361037493f, 0.009264860302209854f, 0.0035319135058671236f, 0.008176364935934544f, 0.006255835760384798f, 0.006094728130847216f, 0.007501496467739344f, 0.005388272926211357f, 0.0070367068983614445f, 0.007998822256922722f, 0.008764436468482018f, 0.006287684198468924f, 0.0069032395258545876f, 0.008075912483036518f, 0.006090731825679541f, 0.008367450907826424f, 0.005319126881659031f, 0.007937949150800705f, 0.00489257974550128f, 0.005988923832774162f, 0.0081204604357481f, 0.0071285320445895195f, 0.005963338539004326f, 0.004493033513426781f, 0.006936715450137854f, 0.009941231459379196f, 0.0032733608968555927f, 0.004502405878156424f, 0.005583086051046848f, 0.004903854336589575f, 0.008089167065918446f, 0.007074758876115084f, 0.006766414735466242f, 0.008433098904788494f, 0.009644391015172005f, 0.005427943542599678f, 0.009563603438436985f, 0.004556438885629177f, 0.008055869489908218f, 0.008679915219545364f, 0.011618289165198803f, 0.004104468040168285f, 0.006301371846348047f, 0.0044442578218877316f, 0.006744819693267345f, 0.010496244765818119f, 0.012288028374314308f, 0.005600146017968655f, 0.008815117180347443f, 0.006209235638380051f, 0.015424676239490509f, 0.005112480837851763f, 0.008811107836663723f, 0.006761190947145224f, 0.012651573866605759f, 0.006547394208610058f, 0.004901160020381212f, 0.009154605679214f, 0.007206837646663189f, 0.006773556582629681f, 0.004570135846734047f, 0.006121499929577112f, 0.00450526550412178f, 0.005796289537101984f, 0.02511613257229328f, 0.006620997563004494f, 0.007383790798485279f, 0.0066080945543944836f, 0.005871608853340149f, 0.009409263730049133f, 0.010589013807475567f, 0.010626341216266155f, 0.005651517771184444f, 0.008144751191139221f, 0.006013066507875919f, 0.015601009130477905f, 0.004558435175567865f, 0.007671280764043331f, 0.0069949207827448845f, 0.004905685782432556f, 0.00575697235763073f, 0.007334728259593248f, 0.005157020408660173f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00014889167505316436f, 8.887290459824726e-05f, 0.00011272893607383594f, 0.00011754345905501395f, 0.00011845186963910237f, 0.00011070183973060921f, 6.783137359889224e-05f, 8.569653437007219e-05f, 0.00016060862981248647f, 0.00012115428398828954f, 0.00011869660374941304f, 9.671336010796949e-05f, 0.00012333669292274863f, 0.00011764237569877878f, 9.112848056247458e-05f, 0.00011733340215869248f, 9.744992712512612e-05f, 9.927609062287956e-05f, 0.00012713448086287826f, 8.658946899231523e-05f, 8.827669080346823e-05f, 0.00012294156476855278f, 7.337213901337236e-05f, 0.00010320148430764675f, 0.00011340588389430195f, 0.00010418918827781454f, 0.0001656823296798393f, 0.00011131451174151152f, 9.157763997791335e-05f, 0.00013132493768353015f, 0.00013438321184366941f, 0.00011024450941476971f, 0.00015450465434696525f, 7.55921719246544e-05f, 0.0001381010952172801f, 9.190883429255337e-05f, 0.00010748991917353123f, 9.659791976446286e-05f, 7.492901931982487e-05f, 8.973359217634425e-05f, 9.453614620724693e-05f, 8.337082545040175e-05f, 9.542296174913645e-05f, 8.55455145938322e-05f, 7.072648440953344e-05f, 0.00011203827307326719f, 0.00010205498983850703f, 0.00010400742758065462f, 0.00014286181249190122f, 0.0001062130686477758f, 0.00014782445214223117f, 0.00011672339314827695f, 0.0001444156951038167f, 9.990115358959883e-05f, 7.229871698655188e-05f, 0.00010900020424742252f, 0.0001174320641439408f, 0.00011887594155268744f, 0.00012726620479952544f, 0.00012434118252713233f, 8.479548705508932e-05f, 0.00013534589379560202f, 0.0001147969815065153f, 0.0001055978937074542f, 7.237901445478201e-05f, 0.0001260824064956978f, 0.00010377561557106674f, 7.532034942414612e-05f, 0.00012635343591682613f, 9.536123980069533e-05f, 8.077586971921846e-05f, 0.0001009305051411502f, 0.00011514306970639154f, 0.00014773791190236807f, 8.068716124398634e-05f, 0.0001227040047524497f, 0.00012012372462777421f, 9.139282337855548e-05f, 0.00012647322728298604f, 0.0001018685070448555f, 8.148639608407393e-05f, 0.00012611648708116263f, 8.800352225080132e-05f, 0.00010372164979344234f, 0.00014665807248093188f, 5.759137275163084e-05f, 0.00011466144496807829f, 0.00013868615496903658f, 8.997697295853868e-05f, 8.945915033109486e-05f, 0.00015084240294527262f, 0.00010210859181825072f, 0.0001294678368140012f, 0.00010613881750032306f, 0.00011720503243850544f, 0.00011996470129815862f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0030042927246540785f, 0.001793251489289105f, 0.002274611499160528f, 0.002371757524088025f, 0.002390087116509676f, 0.0022337092086672783f, 0.0013686815509572625f, 0.0017291596159338951f, 0.0032407138496637344f, 0.002444615587592125f, 0.002395025221630931f, 0.0019514538580551744f, 0.0024886515457183123f, 0.002373753348365426f, 0.0018387638265267015f, 0.00236751907505095f, 0.00196631602011621f, 0.0020031637977808714f, 0.0025652823969721794f, 0.0017471769824624062f, 0.0017812212463468313f, 0.002480678725987673f, 0.0014804814709350467f, 0.002082369290292263f, 0.002288270741701126f, 0.0021022988948971033f, 0.003343089483678341f, 0.002246071584522724f, 0.0018478267593309283f, 0.0026498360093683004f, 0.0027115449775010347f, 0.0022244814317673445f, 0.0031175496987998486f, 0.0015252766897901893f, 0.002786563476547599f, 0.0018545094644650817f, 0.0021689001005142927f, 0.0019491245038807392f, 0.0015118957962840796f, 0.0018106180941686034f, 0.0019075226737186313f, 0.0016822321340441704f, 0.001925416523590684f, 0.0017261123284697533f, 0.001427098293788731f, 0.002260675420984626f, 0.002059235703200102f, 0.0020986313465982676f, 0.0028826238121837378f, 0.0021431362256407738f, 0.0029827584512531757f, 0.0023552104830741882f, 0.002913977485150099f, 0.0020157762337476015f, 0.0014588222838938236f, 0.0021993741393089294f, 0.0023695097770541906f, 0.0023986438754945993f, 0.0025679401587694883f, 0.002508919918909669f, 0.0017109784530475736f, 0.0027309698052704334f, 0.002316339872777462f, 0.002130723325535655f, 0.001460442552343011f, 0.002544053830206394f, 0.002093954011797905f, 0.001519791898317635f, 0.0025495225563645363f, 0.0019241711124777794f, 0.0016298717819154263f, 0.0020365461241453886f, 0.0023233231622725725f, 0.0029810122214257717f, 0.0016280818963423371f, 0.0024758854415267706f, 0.002423821249976754f, 0.0018440976273268461f, 0.002551939571276307f, 0.0020554729271680117f, 0.0016442086780443788f, 0.0025447416119277477f, 0.0017757092136889696f, 0.002092865062877536f, 0.0029592234641313553f, 0.0011620618170127273f, 0.0023136050440371037f, 0.002798368688672781f, 0.0018155289581045508f, 0.0018050805665552616f, 0.0030436536762863398f, 0.002060317201539874f, 0.002612364012748003f, 0.002141637960448861f, 0.0023649288341403008f, 0.00242061261087656f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_37_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001010788619169034f, 0.00010252910578856245f, 0.0001312072854489088f, 0.00011042291589546949f, 0.00011903147969860584f, 9.694818436400965e-05f, 0.00013479305198416114f, 0.00010638077947078273f, 0.00013143295655027032f, 0.00010114331234944984f, 7.180913962656632e-05f, 7.67001838539727e-05f, 9.567038068780676e-05f, 0.0001001025375444442f, 9.925986523739994e-05f, 0.00012187118409201503f, 0.00010362007742514834f, 6.184294761624187e-05f, 0.00010147073771804571f, 8.852638711687177e-05f, 9.211092401528731e-05f, 7.748766074655578e-05f, 0.00011403017560951412f, 0.00010209833271801472f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_37_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003410449717193842f, 0.003459381638094783f, 0.004426997154951096f, 0.0037257226649671793f, 0.004016179591417313f, 0.0032710786908864975f, 0.004547982942312956f, 0.003589338855817914f, 0.004434611648321152f, 0.003412624355405569f, 0.002422875026240945f, 0.002587901195511222f, 0.003227964974939823f, 0.003377507906407118f, 0.0033490757923573256f, 0.004111992660909891f, 0.0034961914643645287f, 0.0020866109989583492f, 0.003423671703785658f, 0.0029869230929762125f, 0.0031078672036528587f, 0.0026144711300730705f, 0.003847433254122734f, 0.00344484718516469f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00014936814841348678f, 0.000152533408254385f, 0.00014067029405850917f, 0.00022868227097205818f, 0.00017582289001438767f, 0.00012331953621469438f, 0.00019975333998445421f, 0.00013703103468287736f, 0.00016756440163590014f, 0.00013021127961110324f, 0.0002630680100992322f, 0.00019521349167916924f, 0.00020555815717671067f, 0.0002072423667414114f, 0.00015353864000644535f, 0.0001182019041152671f, 0.000389212800655514f, 0.00012660691572818905f, 0.00021521648159250617f, 0.00013423631025943905f, 0.0001315888948738575f, 0.00019178027287125587f, 0.00011703053314704448f, 0.00010627080337144434f, 0.0001763604232110083f, 0.0001320522860623896f, 0.00012431983486749232f, 0.00017478808877058327f, 0.00023727498773951083f, 0.00012082684406777844f, 0.00016430532559752464f, 0.00012830535706598312f, 0.00028265980654396117f, 0.00020597997354343534f, 9.801233682082966e-05f, 0.00015432373038493097f, 0.00016612306353636086f, 0.00019335333490744233f, 0.00017318812024313956f, 0.00022569525754079223f, 0.0001374606363242492f, 0.00028842902975156903f, 0.0001590460306033492f, 0.0001544699480291456f, 0.00014258829469326884f, 0.00014978341641835868f, 0.00013168524310458452f, 0.000125314254546538f, 0.00015302389510907233f, 0.00035563326673582196f, 0.00012552061525639147f, 8.787590195424855e-05f, 0.00019580504158511758f, 0.00023394213349092752f, 0.00013421672338154167f, 0.00016971302102319896f, 0.00011760520283132792f, 0.00013936453615315259f, 0.00019840197637677193f, 9.216892794938758e-05f, 0.00020753221178893f, 0.0001635423395782709f, 7.871160050854087e-05f, 0.00024001518613658845f, 0.00015462430019397289f, 0.0003160605556331575f, 0.00010898300388362259f, 0.00014664593618363142f, 0.00021944633044768125f, 9.396457608090714e-05f, 0.00028834326076321304f, 0.00026262179017066956f, 0.00014834436296951026f, 0.00018630101112648845f, 0.00016252652858383954f, 0.00013153209874872118f, 0.0001977022475330159f, 0.00019062659703195095f, 0.00022294593509286642f, 0.00015850915224291384f, 0.0001496536860940978f, 0.0001917664339998737f, 0.00019601899839472026f, 0.00015028017514850944f, 0.0001323290925938636f, 0.00017794744053389877f, 0.00016567717830184847f, 0.0001346963836112991f, 0.00014743860810995102f, 0.0001525637781014666f, 0.00011129509221063927f, 0.0001469061680836603f, 0.00013331662921700627f, 0.00020557148673105985f, 0.0002712187997531146f, 0.00014086347073316574f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0063273184932768345f, 0.006461400538682938f, 0.005958872381597757f, 0.009687108919024467f, 0.007447956129908562f, 0.005223884712904692f, 0.008461663499474525f, 0.005804711487144232f, 0.007098122034221888f, 0.005515822675079107f, 0.011143708601593971f, 0.008269352838397026f, 0.008707558736205101f, 0.00877890270203352f, 0.006503982935100794f, 0.0050070988945662975f, 0.016487272456288338f, 0.0053631397895514965f, 0.00911669060587883f, 0.005686325021088123f, 0.005574179347604513f, 0.008123919367790222f, 0.0049574789591133595f, 0.004501690622419119f, 0.0074707260355353355f, 0.005593808833509684f, 0.005266258027404547f, 0.007404121104627848f, 0.010051101446151733f, 0.005118292756378651f, 0.006960065569728613f, 0.005435086786746979f, 0.011973627842962742f, 0.008725427091121674f, 0.004151857458055019f, 0.006537239532917738f, 0.007037065923213959f, 0.00819055549800396f, 0.007336345501244068f, 0.00956057757139206f, 0.005822909530252218f, 0.012218015268445015f, 0.006737278774380684f, 0.006543433293700218f, 0.006040120031684637f, 0.006344909314066172f, 0.005578260403126478f, 0.005308382213115692f, 0.006482177879661322f, 0.015064824372529984f, 0.005317123606801033f, 0.0037224723491817713f, 0.00829441100358963f, 0.00990991946309805f, 0.005685495678335428f, 0.007189138326793909f, 0.004981822334229946f, 0.005903559736907482f, 0.00840441882610321f, 0.00390432751737535f, 0.008791180327534676f, 0.006927744951099157f, 0.003334267530590296f, 0.010167177766561508f, 0.006549971643835306f, 0.013388501480221748f, 0.004616580903530121f, 0.006212003994733095f, 0.009295869618654251f, 0.003980392124503851f, 0.01221438217908144f, 0.011124806478619576f, 0.006283950060606003f, 0.007891815155744553f, 0.006884714588522911f, 0.005571773275732994f, 0.008374777622520924f, 0.00807504914700985f, 0.009444114752113819f, 0.006714536342769861f, 0.006339413579553366f, 0.008123333565890789f, 0.008303474634885788f, 0.006365952081978321f, 0.00560553465038538f, 0.0075379530899226665f, 0.007018177770078182f, 0.005705814342945814f, 0.006245581898838282f, 0.006462687160819769f, 0.004714522510766983f, 0.006223027594387531f, 0.005647366866469383f, 0.008708123117685318f, 0.011488979682326317f, 0.005967055447399616f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.67940834723413e-05f, 0.0001528380817035213f, 0.00013179356756154448f, 0.00013385695638135076f, 0.00014626167831011117f, 0.00010358022700529546f, 0.00017004902474582195f, 0.00016967054398264736f, 0.00014184240717440844f, 0.00015337869990617037f, 0.00010855380241991952f, 0.0001262323494302109f, 0.00013503417721949518f, 0.00010916510655079037f, 0.00016568707360420376f, 0.00013931942521594465f, 8.580336725572124e-05f, 9.618172771297395e-05f, 0.00014092025230638683f, 0.00014625763287767768f, 0.0001862689241534099f, 0.00016747054178267717f, 0.00013863244384992868f, 0.00012476416304707527f, 0.00010809183731907979f, 0.00016956498438958079f, 0.00010116751946043223f, 8.545505988877267e-05f, 0.00010532265150686726f, 0.00013891693379264325f, 0.00016486551612615585f, 0.00015512543905060738f, 9.890020010061562e-05f, 9.612423309590667e-05f, 0.0001225367741426453f, 0.00015006455942057073f, 0.00010934875172097236f, 0.00019452846026979387f, 0.00014365403330884874f, 8.899678505258635e-05f, 0.00023329992836806923f, 7.288550114026293e-05f, 0.00015642540529370308f, 0.00011324741353746504f, 0.00012045856419717893f, 0.00010161106183659285f, 0.0001251515932381153f, 0.00012591489939950407f, 0.0001717636623652652f, 8.511883061146364e-05f, 0.00012134909047745168f, 0.00015331116446759552f, 8.834907202981412e-05f, 0.0001348713703919202f, 0.00016335527470801026f, 0.0001528569991933182f, 0.00020263086480554193f, 0.0001441673084627837f, 0.0002353706513531506f, 0.00017927501176018268f, 0.00016246704035438597f, 0.00019625206186901778f, 0.00017382822989020497f, 0.00015323724073823541f, 0.00014749786350876093f, 0.00014034956984687597f, 0.00013986293924972415f, 0.0001194923315779306f, 0.00014275680587161332f, 0.00014860968803986907f, 0.00016269684419967234f, 0.00015409631305374205f, 0.00010645113070495427f, 0.00017619914433453232f, 0.00013421055336948484f, 0.00012169519322924316f, 0.00012457351840566844f, 0.0001506548869656399f, 0.00010912805737461895f, 0.00014690768148284405f, 0.00017233315156772733f, 0.00013233177014626563f, 0.00013776772539131343f, 0.00017675975686870515f, 0.00013351028610486537f, 0.0001507853448856622f, 0.00015990079555194825f, 0.00010202596604358405f, 0.00014585786266252398f, 0.0001693792873993516f, 0.000162081079906784f, 0.00013177221990190446f, 0.00010943871166091412f, 0.00017746316734701395f, 0.0001426227536285296f, 0.000134744041133672f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021869027987122536f, 0.0038509771693497896f, 0.003320730058476329f, 0.0033727202098816633f, 0.0036852750927209854f, 0.0026098540984094143f, 0.004284631926566362f, 0.004275095649063587f, 0.003573925234377384f, 0.0038645989261567593f, 0.0027351705357432365f, 0.003180607222020626f, 0.0034023819025605917f, 0.0027505732141435146f, 0.0041747260838747025f, 0.003510355018079281f, 0.0021619403269141912f, 0.0024234382435679436f, 0.00355069013312459f, 0.0036851733457297087f, 0.004693315830081701f, 0.004219663329422474f, 0.0034930456895381212f, 0.0031436141580343246f, 0.0027235306333750486f, 0.004272435791790485f, 0.0025490624830126762f, 0.002153164241462946f, 0.0026537571102380753f, 0.003500213846564293f, 0.0041540260426700115f, 0.003908610437065363f, 0.0024919339921325445f, 0.002421989571303129f, 0.003087491961196065f, 0.003781094215810299f, 0.0027552004903554916f, 0.00490142684429884f, 0.003619571914896369f, 0.0022424031049013138f, 0.005878330208361149f, 0.0018364560091868043f, 0.003941365052014589f, 0.0028534329030662775f, 0.0030351283494383097f, 0.0025602381210774183f, 0.003153375815600157f, 0.003172608558088541f, 0.0043278345838189125f, 0.0021446924656629562f, 0.0030575664713978767f, 0.003862897166982293f, 0.0022260830737650394f, 0.003398279892280698f, 0.004115973133593798f, 0.0038514540065079927f, 0.0051055788062512875f, 0.003632504492998123f, 0.0059305052272975445f, 0.004517094232141972f, 0.0040935929864645f, 0.004944855347275734f, 0.004379854537546635f, 0.003861034754663706f, 0.003716422710567713f, 0.0035363109782338142f, 0.003524049650877714f, 0.0030107826460152864f, 0.003596964757889509f, 0.0037444368936121464f, 0.004099383018910885f, 0.003882680321112275f, 0.0026821906212717295f, 0.00443959329277277f, 0.0033816297072917223f, 0.0030662869103252888f, 0.003138810396194458f, 0.0037959686014801264f, 0.0027496397960931063f, 0.003701552050188184f, 0.0043421839363873005f, 0.003334290813654661f, 0.0034712576307356358f, 0.004453718662261963f, 0.0033639853354543447f, 0.0037992557045072317f, 0.00402893265709281f, 0.002570692216977477f, 0.0036751003935933113f, 0.004267756827175617f, 0.004083868116140366f, 0.003320192452520132f, 0.002757467096671462f, 0.004471442196518183f, 0.0035935870837420225f, 0.0033950714860111475f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_31_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(7.933680171845481e-05f, 5.1800299843307585e-05f, 5.201816384214908e-05f, 6.404783925972879e-05f, 6.494842818938196e-05f, 6.40754951746203e-05f, 6.690341251669452e-05f, 8.89367947820574e-05f, 6.910317461006343e-05f, 6.752441549906507e-05f, 7.538950012531132e-05f, 6.466926424764097e-05f, 6.517148722195998e-05f, 7.779504812788218e-05f, 7.328798528760672e-05f, 8.640468877274543e-05f, 7.580412056995556e-05f, 5.261331898509525e-05f, 7.090393046382815e-05f, 6.436561670852825e-05f, 7.914935849839821e-05f, 9.11390088731423e-05f, 7.576595817226917e-05f, 7.430087134707719e-05f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_31_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0024693228770047426f, 0.0016122614033520222f, 0.0016190423630177975f, 0.0019934605807065964f, 0.0020214910618960857f, 0.0019943213555961847f, 0.002082339022308588f, 0.0027681184001266956f, 0.0021508056670427322f, 0.002101667458191514f, 0.0023464648984372616f, 0.0020128022879362106f, 0.002028433606028557f, 0.00242133648134768f, 0.0022810560185462236f, 0.002689307788386941f, 0.002359369769692421f, 0.0016375662526115775f, 0.0022068533580750227f, 0.0020033512264490128f, 0.0024634888395667076f, 0.0028366614133119583f, 0.002358181867748499f, 0.002312581753358245f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_29_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.946999878389761e-05f, 0.00018064194591715932f, 0.00021899949933867902f, 0.00014626412303186953f, 0.00010136394848814234f, 0.0002137312840204686f, 0.0002608585637062788f, 0.0002365821710554883f, 0.00022402661852538586f, 0.00026155097293667495f, 0.00016881903866305947f, 0.000362120452336967f, 0.00017341203056275845f, 0.00021319724328350276f, 0.0001305877376580611f, 0.0002513025247026235f, 0.00016781047452241182f, 0.0002568814670667052f, 0.00023895310005173087f, 0.00026395873283036053f, 0.0002811214071698487f, 0.00035489880247041583f, 0.00026132873608730733f, 0.0002817233034875244f, 0.0002995013783220202f, 0.00013167156430426985f, 0.00012615857122000307f, 0.00011699224705807865f, 0.0002674365823622793f, 0.00012972358672413975f, 0.0002058823883999139f, 0.00020496334764175117f, 0.00027800226234830916f, 0.00017631925584282726f, 0.00022264692233875394f, 8.062822598731145e-05f, 0.00023083259293343872f, 0.0006002478767186403f, 0.00027928853523917496f, 0.00022180244559422135f, 0.00012536403664853424f, 0.00018996695871464908f, 0.00021821854170411825f, 0.0001469445414841175f, 0.0003820681304205209f, 0.00025063741486519575f, 0.00022565606923308223f, 0.0002396679628873244f, 0.00025235777138732374f, 0.00016571370360907167f, 9.637934999773279e-05f, 0.0003083924821112305f, 0.00022042402997612953f, 0.00012619848712347448f, 0.0003785781154874712f, 0.0002652175899129361f, 0.0003177402832079679f, 0.00026104316930286586f, 0.00020739698084071279f, 0.00020652370585594326f, 0.00015169345715548843f, 0.00026621262077242136f, 0.00020106749434489757f, 0.0001573487970745191f, 0.000473046995466575f, 0.00020730022515635937f, 0.0002687148516997695f, 0.00026804895605891943f, 0.00018891920626629144f, 0.00023516183136962354f, 0.0001391615078318864f, 0.0004184844729024917f, 0.0001951196463778615f, 0.00019697732932399958f, 0.0002162808959838003f, 0.00015107188664842397f, 0.00017609717906452715f, 0.00034175586188212037f, 0.00037770974449813366f, 0.00020617517293430865f, 0.0002143286692444235f, 0.00018979169544763863f, 0.00022999766224529594f, 0.00024758229847066104f, 0.00014763361832592636f, 0.000208923127502203f, 0.0002160781732527539f, 0.0003642571100499481f, 0.00013209943426772952f, 0.00019679628894664347f, 0.00015951386012602597f, 0.00025194164481945336f, 0.0002181347372243181f, 0.00010825111530721188f, 0.0002306685782968998f, 0.0002519482222851366f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_29_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003306866157799959f, 0.006676637567579746f, 0.008094356395304203f, 0.0054060122929513454f, 0.003746473928913474f, 0.007899640128016472f, 0.009641492739319801f, 0.008744223043322563f, 0.008280161768198013f, 0.009667085483670235f, 0.006239655427634716f, 0.013384195044636726f, 0.0064094155095517635f, 0.007879901677370071f, 0.00482660299167037f, 0.009288296103477478f, 0.006202378775924444f, 0.009494497440755367f, 0.008831853978335857f, 0.009756077080965042f, 0.010390420444309711f, 0.01311727799475193f, 0.009658871218562126f, 0.010412666946649551f, 0.011069756001234055f, 0.004866661969572306f, 0.0046628983691334724f, 0.004324105568230152f, 0.009884621016681194f, 0.00479466374963522f, 0.00760954013094306f, 0.007575571537017822f, 0.010275134816765785f, 0.006516868248581886f, 0.008229167200624943f, 0.00298006902448833f, 0.008531714789569378f, 0.02218553237617016f, 0.010322676040232182f, 0.00819795485585928f, 0.004633531905710697f, 0.007021295838057995f, 0.008065491914749146f, 0.005431160796433687f, 0.014121473766863346f, 0.009263712912797928f, 0.008340387605130672f, 0.008858275599777699f, 0.0093272989615798f, 0.006124880630522966f, 0.0035622401628643274f, 0.01139837596565485f, 0.008147007785737514f, 0.004664374049752951f, 0.013992480002343655f, 0.009802605025470257f, 0.011743876151740551f, 0.009648316539824009f, 0.007665520068258047f, 0.0076332432217895985f, 0.005606683436781168f, 0.009839382953941822f, 0.007431578356772661f, 0.005815708544105291f, 0.017484107986092567f, 0.007661943789571524f, 0.009931866079568863f, 0.009907254949212074f, 0.006982570514082909f, 0.008691726252436638f, 0.00514349527657032f, 0.015467444434762001f, 0.007211742457002401f, 0.007280403282493353f, 0.007993875071406364f, 0.005583710037171841f, 0.006508660037070513f, 0.012631507590413094f, 0.013960384763777256f, 0.007620361167937517f, 0.007921719923615456f, 0.007014818023890257f, 0.008500855416059494f, 0.009150794707238674f, 0.005456629674881697f, 0.007721927482634783f, 0.007986382581293583f, 0.01346316747367382f, 0.004882476292550564f, 0.007273712195456028f, 0.005895730573683977f, 0.009311918169260025f, 0.008062394335865974f, 0.004001027904450893f, 0.008525652810931206f, 0.009312161244452f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_27_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001838679745560512f, 0.00017382006626576185f, 0.00013441163173411041f, 0.0002366675907978788f, 0.0002094677183777094f, 0.00016936624888330698f, 0.00013036180462222546f, 0.00013072960427962244f, 7.529683352913707e-05f, 7.752882811473683e-05f, 0.00011886080028489232f, 0.00013980496441945434f, 0.0001391870464431122f, 0.00013332378875929862f, 0.00015305585111491382f, 9.750189201440662e-05f, 0.0001906965480884537f, 0.00016928745026234537f, 0.00021303734683897346f, 0.00013746920740231872f, 0.00022307867766357958f, 0.00025282250135205686f, 0.00015033794625196606f, 0.0001730264921206981f, 0.00013786506315227598f, 0.0002615123230498284f, 0.00019062976934947073f, 0.00016494723968207836f, 0.00011380841897334903f, 0.00018514929979573935f, 0.00016686644812580198f, 0.00019452899869065732f, 0.0002139095449820161f, 0.00011607591295614839f, 0.000163056785822846f, 0.00013990717707201838f, 0.00014910056779626757f, 0.00011120088311145082f, 0.00012666727707255632f, 0.00011440370872151107f, 0.00012167786917416379f, 0.0001330503000644967f, 0.0001621722913114354f, 0.00019422855984885246f, 0.00013453014253173023f, 6.80500888847746e-05f, 7.99495101091452e-05f, 0.0001460318744648248f, 9.779497486306354e-05f, 0.00019469702965579927f, 0.0001800594327505678f, 0.0001482121297158301f, 0.00018713959434535354f, 0.00021565887436736375f, 0.00021373778872657567f, 0.00020935929205734283f, 0.00010049372212961316f, 0.00012775196228176355f, 0.000131668261019513f, 0.00013002743071410805f, 0.00010737156117102131f, 0.00021949331858195364f, 0.00012816577509511262f, 0.00012563931522890925f, 7.215871301013976e-05f, 0.0002020254178205505f, 0.00011811475997092202f, 9.402249997947365e-05f, 0.00016438557941000909f, 0.00012559653259813786f, 0.00014620606089010835f, 7.489432027796283e-05f, 0.000139206851599738f, 0.0002080204285448417f, 0.0001322931348113343f, 0.00019160112424287945f, 0.0002988856576848775f, 8.071238698903471e-05f, 0.00010760482109617442f, 0.00016319143469445407f, 0.00013372254034038633f, 0.00021213351283222437f, 0.00010355745325796306f, 0.00013668049359694123f, 0.0001614428183529526f, 0.00017566110182087868f, 0.00011349537817295641f, 0.00018218533659819514f, 0.00016371843230444938f, 0.00010239308176096529f, 0.00015248243289534003f, 0.00013820073218084872f, 0.00017066638974938542f, 0.00020325709192547947f, 0.0001279056741623208f, 0.00011325388913974166f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_27_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005157727748155594f, 0.004875871352851391f, 0.003770414972677827f, 0.006638823077082634f, 0.00587583240121603f, 0.004750935826450586f, 0.0036568124778568745f, 0.003667129436507821f, 0.0021121706813573837f, 0.002174780936911702f, 0.003334194654598832f, 0.0039217048324644566f, 0.0039043715223670006f, 0.0037398997228592634f, 0.004293408710509539f, 0.002735050627961755f, 0.005349277518689632f, 0.004748725797981024f, 0.005975964944809675f, 0.003856183961033821f, 0.006257636938244104f, 0.00709198834374547f, 0.004217168316245079f, 0.004853610415011644f, 0.003867288352921605f, 0.007335749454796314f, 0.005347404628992081f, 0.0046269772574305534f, 0.003192469011992216f, 0.005193670280277729f, 0.004680813290178776f, 0.005456782877445221f, 0.006000431254506111f, 0.003256075084209442f, 0.004573947750031948f, 0.00392457190901041f, 0.0041824583895504475f, 0.0031193243339657784f, 0.0035531760659068823f, 0.0032091676257550716f, 0.003413217142224312f, 0.003732228185981512f, 0.004549136385321617f, 0.005448354873806238f, 0.0037737395614385605f, 0.0019088905537500978f, 0.002242684131488204f, 0.004096377640962601f, 0.002743271877989173f, 0.00546149630099535f, 0.005050893407315016f, 0.004157536197453737f, 0.00524950074031949f, 0.006049502175301313f, 0.005995613522827625f, 0.005872791167348623f, 0.002818975131958723f, 0.003583603072911501f, 0.003693460253998637f, 0.0036474326625466347f, 0.0030119072180241346f, 0.006157063413411379f, 0.0035952110774815083f, 0.0035243406891822815f, 0.002024142537266016f, 0.005667066667228937f, 0.003313267370685935f, 0.002637449186295271f, 0.004611222073435783f, 0.0035231406800448895f, 0.004101263824850321f, 0.002100879792124033f, 0.003904926823452115f, 0.005835234187543392f, 0.0037109884433448315f, 0.005374652333557606f, 0.008384117856621742f, 0.002264083828777075f, 0.0030184504576027393f, 0.004577724728733301f, 0.0037510853726416826f, 0.005950611550360918f, 0.002904916647821665f, 0.003834059461951256f, 0.004528673831373453f, 0.004927514586597681f, 0.0031836878042668104f, 0.005110527388751507f, 0.004592507611960173f, 0.002872254466637969f, 0.004277323838323355f, 0.0038767042569816113f, 0.0047874064184725285f, 0.005701616872102022f, 0.0035879146307706833f, 0.0031769138295203447f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_26_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.379693801747635e-05f, 0.00011589877249207348f, 0.00010584280971670523f, 0.00014072666817810386f, 8.91255695023574e-05f, 0.0001251988869626075f, 0.00010738251148723066f, 0.00010287849727319553f, 0.0001043882075464353f, 8.981209248304367e-05f, 8.665172208566219e-05f, 0.0001264015445485711f, 0.0001013853179756552f, 0.0001156824073405005f, 9.315105126006529e-05f, 9.523649350740016e-05f, 9.071284875972196e-05f, 0.00010555300104897469f, 0.0001387594238622114f, 0.00011888876179000363f, 0.00010910760465776548f, 9.382123243995011e-05f, 7.950081635499373e-05f, 9.301745740231127e-05f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_26_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 24,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016981291119009256f, 0.00209826766513288f, 0.0019162113312631845f, 0.002547759562730789f, 0.0016135572222992778f, 0.002266639843583107f, 0.0019440866308286786f, 0.0018625445663928986f, 0.00188987678848207f, 0.0016259861877188087f, 0.0015687699196860194f, 0.0022884132340550423f, 0.0018355115316808224f, 0.0020943505223840475f, 0.0016864357749000192f, 0.001724191359244287f, 0.0016422937624156475f, 0.0019109646091237664f, 0.002512143924832344f, 0.002152399392798543f, 0.001975318184122443f, 0.0016985689289867878f, 0.0014393076999112964f, 0.0016840171301737428f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_24_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003490637172944844f, 0.0006226297118701041f, 0.0002743686782196164f, 0.0005805440014228225f, 0.0006724314298480749f, 0.0003873146197292954f, 0.00022657914087176323f, 0.0005490295006893575f, 0.0006306810537353158f, 0.0009225341491401196f, 0.0004970444133505225f, 0.00036430376349017024f, 0.000318280013743788f, 0.0003542272897902876f, 0.0009472741512581706f, 0.000310191186144948f, 0.0003501390456221998f, 0.00040909278322942555f, 0.0008177982526831329f, 0.00036680116318166256f, 0.0004165370191913098f, 0.00039263840881176293f, 0.0007649771869182587f, 0.0005838179495185614f, 0.0009378187241964042f, 0.00028053869027644396f, 0.0008142348960973322f, 0.00047431403072550893f, 0.0004987137508578598f, 0.000489905069116503f, 0.0005528611363843083f, 0.000155598419951275f, 0.00045613947440870106f, 0.0002995508548337966f, 0.0004919803468510509f, 0.0006526862853206694f, 0.0007902536308392882f, 0.0004425451043061912f, 0.0002707020321395248f, 0.0004152815090492368f, 0.00033696924219839275f, 0.00046983451466076076f, 0.00035813028807751834f, 0.0005589158972725272f, 0.00035931746242567897f, 0.0006035493570379913f, 0.00038183596916496754f, 0.0002654145355336368f, 0.0005752751603722572f, 0.00030214915750548244f, 0.00036037765676155686f, 0.00026371877174824476f, 0.0015978048322722316f, 0.00022382430324796587f, 0.0005336849135346711f, 0.0004904686356894672f, 0.0005768064875155687f, 0.0006554853171110153f, 0.00033563870238140225f, 0.0004659139667637646f, 0.001047445461153984f, 0.0007014462607912719f, 0.001029566046781838f, 0.0006208459381014109f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_24_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0054702674970030785f, 0.009757391177117825f, 0.00429970258846879f, 0.009097854606807232f, 0.010537846945226192f, 0.006069707218557596f, 0.003550780238583684f, 0.008603982627391815f, 0.009883565828204155f, 0.014457270503044128f, 0.0077893114648759365f, 0.005709098186343908f, 0.004987848456948996f, 0.0055511873215436935f, 0.014844977296888828f, 0.004861086141318083f, 0.005487119313329458f, 0.00641099875792861f, 0.012815927155315876f, 0.005748235620558262f, 0.0065276590175926685f, 0.006153137888759375f, 0.011988154612481594f, 0.009149162098765373f, 0.014696799218654633f, 0.004396394360810518f, 0.012760085053741932f, 0.007433097809553146f, 0.007815471850335598f, 0.007677428890019655f, 0.008664029650390148f, 0.002438422990962863f, 0.007148279342800379f, 0.004694338887929916f, 0.007709951139986515f, 0.0102284150198102f, 0.012384268455207348f, 0.0069352383725345135f, 0.004242241382598877f, 0.006507983896881342f, 0.005280732177197933f, 0.007362897973507643f, 0.005612352397292852f, 0.008758915588259697f, 0.005630956497043371f, 0.009458377957344055f, 0.005983849987387657f, 0.004159379750490189f, 0.009015285409986973f, 0.00473505724221468f, 0.005647571291774511f, 0.004132804926484823f, 0.025039611384272575f, 0.003507608314976096f, 0.008363514207303524f, 0.0076862601563334465f, 0.009039283730089664f, 0.010272279381752014f, 0.005259880796074867f, 0.007301458157598972f, 0.016414787620306015f, 0.010992545634508133f, 0.01613459549844265f, 0.009729436598718166f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_21_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000468509882921353f, 0.00022249066387303174f, 0.000305414228932932f, 0.0004218710237182677f, 0.0002178022696170956f, 0.0004341299645602703f, 0.00028514806763269007f, 0.00020224833860993385f, 0.00018951036327052861f, 0.00034396149567328393f, 0.0002186697965953499f, 0.0004076736222486943f, 0.0002799852518364787f, 0.00034239451633766294f, 0.0003297276562079787f, 0.00039721274515613914f, 0.00035487671266309917f, 0.00028672540793195367f, 0.00021567432850133628f, 0.0002986956969834864f, 0.0002682673220988363f, 0.00031466272776015103f, 0.0002523352741263807f, 0.00033733740565367043f, 0.0002077477693092078f, 0.00030983961187303066f, 0.00027476143441163003f, 0.00021085933258291334f, 0.0003147181705571711f, 0.00034628980210982263f, 0.0003867661871481687f, 0.0003494849079288542f, 0.000192114501260221f, 0.00023834832245483994f, 0.00037590667488984764f, 0.00018241271027363837f, 0.0002808691933751106f, 0.00016020567272789776f, 0.00025486209779046476f, 0.00032009033020585775f, 0.00032194473897106946f, 0.00020519140525721014f, 0.00034451394458301365f, 0.0003506063367240131f, 0.00023789542319718748f, 0.0002654523414094001f, 0.0003074769920203835f, 0.00043559714686125517f, 0.00027974843396805227f, 0.0003198085760232061f, 0.0002833425533026457f, 0.0003063430485781282f, 0.0002436151698930189f, 0.00023754127323627472f, 0.0003649516438599676f, 0.0005986230098642409f, 0.0002678904274944216f, 0.00022338143025990576f, 0.0003697171632666141f, 0.00044405931839719415f, 0.00026240531587973237f, 0.00026048513245768845f, 0.00034119136398658156f, 0.00017907709116116166f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_21_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004906991496682167f, 0.0023302813060581684f, 0.003198790829628706f, 0.0044185142032802105f, 0.00228117685765028f, 0.0045469095930457115f, 0.0029865307733416557f, 0.0021182710770517588f, 0.0019848584197461605f, 0.003602519864216447f, 0.0022902630735188723f, 0.004269815981388092f, 0.00293245748616755f, 0.0035861078649759293f, 0.0034534400328993797f, 0.004160252865403891f, 0.0037168418057262897f, 0.003003051271662116f, 0.0022588896099478006f, 0.0031284233555197716f, 0.0028097282629460096f, 0.0032956558279693127f, 0.0026428622659295797f, 0.0035331416875123978f, 0.0021758698858320713f, 0.003245140425860882f, 0.0028777450788766146f, 0.0022084591910243034f, 0.0032962365075945854f, 0.0036269056145101786f, 0.004050839692354202f, 0.0036603701300919056f, 0.0020121331326663494f, 0.0024963682517409325f, 0.003937101457268f, 0.001910520251840353f, 0.002941715531051159f, 0.0016779323341324925f, 0.0026693271938711405f, 0.003352502593770623f, 0.0033719248604029417f, 0.0021490955259650946f, 0.003608305938541889f, 0.003672115271911025f, 0.002491624793037772f, 0.0027802453842014074f, 0.003220395417883992f, 0.004562276415526867f, 0.0029299771413207054f, 0.0033495514653623104f, 0.002967620501294732f, 0.0032085187267512083f, 0.0025515311863273382f, 0.0024879155680537224f, 0.0038223627489060163f, 0.006269745994359255f, 0.002805780852213502f, 0.002339610829949379f, 0.00387227488681674f, 0.004650905728340149f, 0.0027483319863677025f, 0.002728220773860812f, 0.0035735066048800945f, 0.0018755843630060554f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_17_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00023435079492628574f, 0.00018155357975047082f, 0.00020313546701800078f, 0.00036327008274383843f, 0.0002453649649396539f, 0.00029520553653128445f, 0.0002649530360940844f, 0.00032074347836896777f, 0.00025488450773991644f, 0.000279587518889457f, 0.0002514767402317375f, 0.0002388852008152753f, 0.0002445672289468348f, 0.00020894902991130948f, 0.00021879056293983012f, 0.00045221790787763894f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_17_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0038104685954749584f, 0.0029520029202103615f, 0.0033029173500835896f, 0.005906654987484217f, 0.003989555407315493f, 0.004799947142601013f, 0.0043080514296889305f, 0.005215185694396496f, 0.0041443402878940105f, 0.004546003416180611f, 0.004088930785655975f, 0.003884196514263749f, 0.0039765844121575356f, 0.0033974440302699804f, 0.0035574643407016993f, 0.007352917920798063f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001753582851961255f, 0.0007848356035538018f, 0.00048680626787245274f, 0.0006367580499500036f, 0.0012401145650073886f, 0.00027729716384783387f, 0.000742071308195591f, 0.0013035967713221908f, 0.0004914581077173352f, 0.0014965281588956714f, 0.001042736228555441f, 0.0006329853204078972f, 0.0006875276449136436f, 0.001009719679132104f, 0.0002776132896542549f, 0.0004056294565089047f, 0.0005613952525891364f, 0.0008261182811111212f, 0.0007975423359312117f, 0.0005699400207959116f, 0.0008295270963571966f, 0.0010324037866666913f, 0.0007516051991842687f, 0.0013051554560661316f, 0.0006401427090167999f, 0.0006712193717248738f, 0.0009433191153220832f, 0.0007630485924892128f, 0.0010953932069242f, 0.0011317452881485224f, 0.0006605290109291673f, 0.0006404794403351843f, 0.0004582286055665463f, 0.0007251044735312462f, 0.0008287234813906252f, 0.00047457506298087537f, 0.0016443576896563172f, 0.000621542742010206f, 0.000579823215957731f, 0.0008972593350335956f, 0.0003886436752509326f, 0.001266798353753984f, 0.0006869005737826228f, 0.00026881255325861275f, 0.0006388885085470974f, 0.0004732178640551865f, 0.00041885519749484956f, 0.001636386034078896f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03134153038263321f, 0.014027252793312073f, 0.008700617589056492f, 0.01138068363070488f, 0.02216438762843609f, 0.00495609175413847f, 0.013262932188808918f, 0.023298995569348335f, 0.00878375954926014f, 0.02674723044037819f, 0.01863667368888855f, 0.011313254944980145f, 0.012288081459701061f, 0.01804657280445099f, 0.004961742088198662f, 0.007249756250530481f, 0.010033736005425453f, 0.014765091240406036f, 0.014254357665777206f, 0.010186455212533474f, 0.014826016500592232f, 0.018452001735568047f, 0.013433330692350864f, 0.023326853290200233f, 0.011441177688539028f, 0.01199660636484623f, 0.016859805211424828f, 0.01363785658031702f, 0.01957780309021473f, 0.020227517932653427f, 0.011805539019405842f, 0.011447195895016193f, 0.0081898532807827f, 0.012959687039256096f, 0.014811654575169086f, 0.008482011035084724f, 0.029389364644885063f, 0.011108743026852608f, 0.010363096371293068f, 0.016036584973335266f, 0.006946171633899212f, 0.02264130301773548f, 0.012276873923838139f, 0.0048044477589428425f, 0.011418761685490608f, 0.00845775380730629f, 0.00748613802716136f, 0.029246889054775238f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00017695649876259267f, 0.0005025022546760738f, 0.00013014480646234006f, 0.0002598148712422699f, 0.00024064694298431277f, 0.00030403686105273664f, 0.00037359129055403173f, 0.00022186158457770944f, 0.0003596720052883029f, 0.00035102860420010984f, 0.00022121994697954506f, 0.0003954873245675117f, 0.0002677832671906799f, 0.00029179322882555425f, 0.00031328736804425716f, 0.00019048676767852157f, 0.00025947255198843777f, 0.00020209279318805784f, 0.00020580650016199797f, 0.00047863664804026484f, 0.00034489939571358263f, 5.183097528060898e-05f, 0.00014357139298226684f, 0.000210141995921731f, 0.0003699645458254963f, 0.00015935311967041343f, 0.0002973532536998391f, 0.00022646106663160026f, 0.0002089138433802873f, 0.00021262411610223353f, 0.0001869115949375555f, 0.0005067480378784239f, 0.00033780556987039745f, 0.00041489212890155613f, 0.00029248330974951386f, 0.0002464350254740566f, 0.00021658190235029906f, 0.0004018997715320438f, 0.0003084209456574172f, 0.0003751789918169379f, 0.0003805176238529384f, 0.000175361055880785f, 0.00035513314651325345f, 0.0004980486119166017f, 0.0003084728086832911f, 0.00047484191600233316f, 0.0006016995175741613f, 0.00015346230065915734f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00306734093464911f, 0.00871030893176794f, 0.0022559131029993296f, 0.004503597505390644f, 0.004171343054622412f, 0.005270135588943958f, 0.0064757829532027245f, 0.003845720086246729f, 0.006234508007764816f, 0.006084684282541275f, 0.003834597999230027f, 0.006855326239019632f, 0.004641720559448004f, 0.005057905800640583f, 0.005430482793599367f, 0.0033018728718161583f, 0.0044976635836064816f, 0.003503050422295928f, 0.0035674232058227062f, 0.008296625688672066f, 0.005978441331535578f, 0.0008984314044937491f, 0.002488647820428014f, 0.0036425741855055094f, 0.006412917282432318f, 0.0027622063644230366f, 0.005154282785952091f, 0.0039254468865692616f, 0.0036212855484336615f, 0.003685598960146308f, 0.0032399012707173824f, 0.00878390483558178f, 0.0058554778806865215f, 0.007191686425358057f, 0.005069867707788944f, 0.004271672572940588f, 0.0037542027421295643f, 0.006966478656977415f, 0.0053461287170648575f, 0.006503304000943899f, 0.006595843005925417f, 0.0030396857764571905f, 0.006155832204967737f, 0.008633109740912914f, 0.005347027909010649f, 0.008230848237872124f, 0.01042978186160326f, 0.002660095691680908f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002637974394019693f, 0.00032117238151840866f, 0.0003086063952650875f, 0.00027033634250983596f, 0.00024605102953501046f, 0.00038511669845320284f, 0.00029971188632771373f, 0.0002790583821479231f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0016813657712191343f, 0.002047056332230568f, 0.001966964453458786f, 0.0017230426892638206f, 0.0015682554803788662f, 0.002454618224874139f, 0.0019102736841887236f, 0.001778634381480515f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_9_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006546095246449113f, 0.0012638011248782277f, 0.0005703847855329514f, 0.0008081647101789713f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_9_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 4,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00674839224666357f, 0.013028568588197231f, 0.005880116019397974f, 0.00833139754831791f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002921194536611438f, 0.0014276013243943453f, 0.00335462624207139f, 0.000309582072077319f, 0.002784295938909054f, 0.0010677071986719966f, 0.0004087032575625926f, 0.0003498844744171947f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020750094205141068f, 0.010140666738152504f, 0.023828884586691856f, 0.002199051436036825f, 0.019777663052082062f, 0.007584234233945608f, 0.002903138054534793f, 0.0024853311479091644f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000211369784665294f, 0.0005132610676810145f, 0.00036175837158225477f, 0.0012070987140759826f, 0.0007704274612478912f, 0.0004900377243757248f, 0.0008149809436872602f, 0.0013526168186217546f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019709423650056124f, 0.0047859628684818745f, 0.0033732582814991474f, 0.011255733668804169f, 0.007183941081166267f, 0.00456941407173872f, 0.00759938545525074f, 0.012612634338438511f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_bias_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00031571416184306145f, 0.00024308175488840789f, 4.34275898442138e-05f, 0.0004044623929075897f, 0.00015775336942169815f, 0.0002155525580747053f, 6.0937680245842785e-05f, 0.00037871176027692854f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 8,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08050710707902908f, 0.061985842883586884f, 0.011074034497141838f, 0.10313790291547775f, 0.04022710770368576f, 0.054965898394584656f, 0.015539107844233513f, 0.09657149016857147f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10724300891160965f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14077982306480408f),
    AI_PACK_INTQ_ZP(-104)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_7_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0970022976398468f),
    AI_PACK_INTQ_ZP(-105)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_9_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1568947434425354f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05769051983952522f),
    AI_PACK_INTQ_ZP(-114)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.055950772017240524f),
    AI_PACK_INTQ_ZP(-95)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06150183081626892f),
    AI_PACK_INTQ_ZP(-97)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_17_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08666469901800156f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_20_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09547802805900574f),
    AI_PACK_INTQ_ZP(-19)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_21_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06381108611822128f),
    AI_PACK_INTQ_ZP(-105)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_24_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05523545667529106f),
    AI_PACK_INTQ_ZP(-95)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_26_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.035649027675390244f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_27_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.027055826038122177f),
    AI_PACK_INTQ_ZP(-98)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_29_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03212897107005119f),
    AI_PACK_INTQ_ZP(-85)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_31_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025511162355542183f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_32_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03968812897801399f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023606864735484123f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029637986794114113f),
    AI_PACK_INTQ_ZP(-94)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_37_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.037036824971437454f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_38_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04955964535474777f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021849945187568665f),
    AI_PACK_INTQ_ZP(-106)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_41_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04172040522098541f),
    AI_PACK_INTQ_ZP(-76)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_43_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.058767255395650864f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_44_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07315566390752792f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_47_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08474089205265045f),
    AI_PACK_INTQ_ZP(-18)))

/* Int quant #92 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_48_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02483389526605606f),
    AI_PACK_INTQ_ZP(-107)))

/* Int quant #93 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_51_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03973064944148064f),
    AI_PACK_INTQ_ZP(-103)))

/* Int quant #94 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_53_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0455172024667263f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #95 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_54_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.032391566783189774f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #96 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_55_output_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10756264626979828f),
    AI_PACK_INTQ_ZP(32)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1324, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1324, 1324),
  1, &conv2d_55_scratch0_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 2816, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2816, 2816),
  1, &conv2d_54_scratch0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1152, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1152, 1152),
  1, &conv2d_53_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_51_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 5, 5), AI_STRIDE_INIT(4, 1, 1, 128, 640),
  1, &conv2d_51_scratch1_array, &conv2d_51_scratch1_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_51_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 4737, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4737, 4737),
  1, &conv2d_51_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_48_scratch1_array, &conv2d_48_scratch1_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1632, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1632, 1632),
  1, &conv2d_48_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_43_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 624, 1, 1), AI_STRIDE_INIT(4, 1, 1, 624, 624),
  1, &conv2d_43_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_41_scratch1_array, &conv2d_41_scratch1_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 3553, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3553, 3553),
  1, &conv2d_41_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_39_scratch1_array, &conv2d_39_scratch1_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1056, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1056, 1056),
  1, &conv2d_39_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 624, 1, 1), AI_STRIDE_INIT(4, 1, 1, 624, 624),
  1, &conv2d_37_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_35_scratch1_array, &conv2d_35_scratch1_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 3553, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3553, 3553),
  1, &conv2d_35_scratch0_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_33_scratch1_array, &conv2d_33_scratch1_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1056, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1056, 1056),
  1, &conv2d_33_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_31_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 624, 1, 1), AI_STRIDE_INIT(4, 1, 1, 624, 624),
  1, &conv2d_31_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_29_scratch1_array, &conv2d_29_scratch1_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 3553, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3553, 3553),
  1, &conv2d_29_scratch0_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_27_scratch1_array, &conv2d_27_scratch1_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1056, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1056, 1056),
  1, &conv2d_27_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 496, 1, 1), AI_STRIDE_INIT(4, 1, 1, 496, 496),
  1, &conv2d_26_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 1, 1, 64, 640),
  1, &conv2d_24_scratch1_array, &conv2d_24_scratch1_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 2369, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2369, 2369),
  1, &conv2d_24_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_21_scratch1_array, &conv2d_21_scratch1_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 736, 1, 1), AI_STRIDE_INIT(4, 1, 1, 736, 736),
  1, &conv2d_21_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 352, 1, 1), AI_STRIDE_INIT(4, 1, 1, 352, 352),
  1, &conv2d_17_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 20, 20), AI_STRIDE_INIT(4, 1, 1, 48, 960),
  1, &conv2d_15_scratch1_array, &conv2d_15_scratch1_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1777, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1777, 1777),
  1, &conv2d_15_scratch0_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 20, 20), AI_STRIDE_INIT(4, 1, 1, 48, 960),
  1, &conv2d_13_scratch1_array, &conv2d_13_scratch1_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_13_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 20, 20), AI_STRIDE_INIT(4, 1, 1, 8, 160),
  1, &conv2d_11_scratch1_array, &conv2d_11_scratch1_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 832, 1, 1), AI_STRIDE_INIT(4, 1, 1, 832, 832),
  1, &conv2d_11_scratch0_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 72, 1, 1), AI_STRIDE_INIT(4, 1, 1, 72, 72),
  1, &conv2d_9_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 40, 40), AI_STRIDE_INIT(4, 1, 1, 8, 320),
  1, &conv2d_7_scratch1_array, &conv2d_7_scratch1_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 297, 1, 1), AI_STRIDE_INIT(4, 1, 1, 297, 297),
  1, &conv2d_7_scratch0_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_4_scratch1_array, &conv2d_4_scratch1_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 1, 1, 112, 112),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch1, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_2_scratch1_array, &conv2d_2_scratch1_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 292, 1, 1), AI_STRIDE_INIT(4, 1, 1, 292, 292),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 1), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &conv2d_55_bias_array, &conv2d_55_bias_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 30), AI_STRIDE_INIT(4, 1, 256, 256, 256),
  1, &conv2d_55_weights_array, &conv2d_55_weights_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_54_bias_array, &conv2d_54_bias_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 256), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_54_weights_array, &conv2d_54_weights_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_53_bias_array, &conv2d_53_bias_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 64), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_53_weights_array, &conv2d_53_weights_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_51_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_51_bias_array, &conv2d_51_bias_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_51_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_51_weights_array, &conv2d_51_weights_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_48_bias_array, &conv2d_48_bias_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 88, 1, 1, 128), AI_STRIDE_INIT(4, 1, 88, 88, 88),
  1, &conv2d_48_weights_array, &conv2d_48_weights_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_43_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &conv2d_43_bias_array, &conv2d_43_bias_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_43_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 24), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &conv2d_43_weights_array, &conv2d_43_weights_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_41_bias_array, &conv2d_41_bias_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 3, 3, 1), AI_STRIDE_INIT(4, 1, 96, 288, 864),
  1, &conv2d_41_weights_array, &conv2d_41_weights_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_39_bias_array, &conv2d_39_bias_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 96), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &conv2d_39_weights_array, &conv2d_39_weights_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &conv2d_37_bias_array, &conv2d_37_bias_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 24), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &conv2d_37_weights_array, &conv2d_37_weights_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_35_bias_array, &conv2d_35_bias_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 3, 3, 1), AI_STRIDE_INIT(4, 1, 96, 288, 864),
  1, &conv2d_35_weights_array, &conv2d_35_weights_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_33_bias_array, &conv2d_33_bias_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 96), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &conv2d_33_weights_array, &conv2d_33_weights_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_31_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &conv2d_31_bias_array, &conv2d_31_bias_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_31_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 24), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &conv2d_31_weights_array, &conv2d_31_weights_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_29_bias_array, &conv2d_29_bias_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 96, 3, 3, 1), AI_STRIDE_INIT(4, 1, 96, 288, 864),
  1, &conv2d_29_weights_array, &conv2d_29_weights_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &conv2d_27_bias_array, &conv2d_27_bias_intq)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 96), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &conv2d_27_weights_array, &conv2d_27_weights_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &conv2d_26_bias_array, &conv2d_26_bias_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 24), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_26_weights_array, &conv2d_26_weights_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_24_bias_array, &conv2d_24_bias_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 3, 1), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_24_weights_array, &conv2d_24_weights_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_21_bias_array, &conv2d_21_bias_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 64), AI_STRIDE_INIT(4, 1, 24, 24, 24),
  1, &conv2d_21_weights_array, &conv2d_21_weights_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_17_bias_array, &conv2d_17_bias_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 1, 16), AI_STRIDE_INIT(4, 1, 48, 48, 48),
  1, &conv2d_17_weights_array, &conv2d_17_weights_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_15_bias_array, &conv2d_15_bias_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 48, 3, 3, 1), AI_STRIDE_INIT(4, 1, 48, 144, 432),
  1, &conv2d_15_weights_array, &conv2d_15_weights_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2d_13_bias_array, &conv2d_13_bias_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 48), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &conv2d_13_weights_array, &conv2d_13_weights_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_11_bias_array, &conv2d_11_bias_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 4, 3, 3, 8), AI_STRIDE_INIT(4, 1, 4, 12, 36),
  1, &conv2d_11_weights_array, &conv2d_11_weights_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv2d_9_bias_array, &conv2d_9_bias_intq)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 4), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &conv2d_9_weights_array, &conv2d_9_weights_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_7_bias_array, &conv2d_7_bias_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 3, 1), AI_STRIDE_INIT(4, 1, 8, 24, 72),
  1, &conv2d_7_weights_array, &conv2d_7_weights_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_4_bias_array, &conv2d_4_bias_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 8), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &conv2d_4_weights_array, &conv2d_4_weights_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_2_bias_array, &conv2d_2_bias_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1, 1, 3, 9),
  1, &conv2d_2_weights_array, &conv2d_2_weights_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  image_input_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 160, 160), AI_STRIDE_INIT(4, 4, 4, 4, 640),
  1, &image_input_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 160, 160), AI_STRIDE_INIT(4, 1, 1, 1, 160),
  1, &conversion_0_output_array, &conversion_0_output_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_2_output_array, &conv2d_2_output_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_4_output_array, &conv2d_4_output_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 40, 40), AI_STRIDE_INIT(4, 1, 1, 8, 320),
  1, &conv2d_7_output_array, &conv2d_7_output_intq)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 40, 40), AI_STRIDE_INIT(4, 1, 1, 4, 160),
  1, &conv2d_9_output_array, &conv2d_9_output_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 20, 20), AI_STRIDE_INIT(4, 1, 1, 8, 160),
  1, &conv2d_11_output_array, &conv2d_11_output_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 20, 20), AI_STRIDE_INIT(4, 1, 1, 48, 960),
  1, &conv2d_13_output_array, &conv2d_13_output_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 20, 20), AI_STRIDE_INIT(4, 1, 1, 48, 960),
  1, &conv2d_15_output_array, &conv2d_15_output_intq)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 20, 20), AI_STRIDE_INIT(4, 1, 1, 16, 320),
  1, &conv2d_17_output_array, &conv2d_17_output_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  concat_20_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 20, 20), AI_STRIDE_INIT(4, 1, 1, 24, 480),
  1, &concat_20_output_array, &concat_20_output_intq)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_21_output_array, &conv2d_21_output_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 1, 1, 64, 640),
  1, &conv2d_24_output_array, &conv2d_24_output_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &conv2d_26_output_array, &conv2d_26_output_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_27_output_array, &conv2d_27_output_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_29_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_29_output_array, &conv2d_29_output_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_31_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &conv2d_31_output_array, &conv2d_31_output_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_32_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &eltwise_32_output_array, &eltwise_32_output_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_33_output_array, &conv2d_33_output_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_35_output_array, &conv2d_35_output_intq)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &conv2d_37_output_array, &conv2d_37_output_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &eltwise_38_output_array, &eltwise_38_output_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_39_output_array, &conv2d_39_output_intq)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_41_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 10, 10), AI_STRIDE_INIT(4, 1, 1, 96, 960),
  1, &conv2d_41_output_array, &conv2d_41_output_intq)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_43_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &conv2d_43_output_array, &conv2d_43_output_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_44_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 10, 10), AI_STRIDE_INIT(4, 1, 1, 24, 240),
  1, &eltwise_44_output_array, &eltwise_44_output_intq)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  concat_47_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 88, 10, 10), AI_STRIDE_INIT(4, 1, 1, 88, 880),
  1, &concat_47_output_array, &concat_47_output_intq)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_48_output_array, &conv2d_48_output_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_51_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 5, 5), AI_STRIDE_INIT(4, 1, 1, 128, 640),
  1, &conv2d_51_output_array, &conv2d_51_output_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 5), AI_STRIDE_INIT(4, 1, 1, 64, 320),
  1, &conv2d_53_output_array, &conv2d_53_output_intq)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 5, 5), AI_STRIDE_INIT(4, 1, 1, 256, 1280),
  1, &conv2d_54_output_array, &conv2d_54_output_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_55_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 5, 5), AI_STRIDE_INIT(4, 1, 1, 30, 150),
  1, &conv2d_55_output_array, &conv2d_55_output_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  conversion_56_output, AI_STATIC,
  0x0, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 5, 5), AI_STRIDE_INIT(4, 4, 4, 120, 600),
  1, &conversion_56_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &image_input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_0_layer, 0,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &conv2d_2_layer, AI_STATIC,
  .tensors = &conversion_0_chain, 
)


AI_STATIC_CONST ai_i8 conv2d_2_nl_params_data[] = { -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -104, -104, -104, -104, -104, -104, -103, -103, -103, -101, -99, -97, -96, -94, -92, -90, -88, -86, -84, -82, -81, -79, -77, -75, -73, -71, -69, -67, -66, -64, -62, -60, -58, -56, -54, -52, -51, -49, -47, -45, -43, -41, -39, -38, -36, -34, -32, -30, -28, -26, -24, -23, -21, -19, -17, -15, -13, -11, -9, -8, -6, -4, -2, 0, 2, 4, 6, 7, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 35, 37, 39, 41, 43, 45, 47, 49, 50, 52, 54, 56, 58, 60, 62, 64, 65, 67, 69, 71, 73, 75, 77, 79, 80, 82, 84, 86, 88, 90, 92, 93, 95, 97, 99, 101, 103, 105, 107, 108, 110, 112, 114, 116, 118, 120, 122, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_2_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_2_nl_params_data, conv2d_2_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_2_scratch0, &conv2d_2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_4_layer, AI_STATIC,
  .tensors = &conv2d_2_chain, 
  .groups = 1, 
  .nl_params = &conv2d_2_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_4_nl_params_data[] = { -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -105, -104, -104, -104, -102, -100, -98, -97, -95, -93, -91, -89, -87, -85, -84, -82, -80, -78, -76, -74, -72, -71, -69, -67, -65, -63, -61, -59, -58, -56, -54, -52, -50, -48, -47, -45, -43, -41, -39, -37, -35, -34, -32, -30, -28, -26, -24, -22, -21, -19, -17, -15, -13, -11, -9, -8, -6, -4, -2, 0, 2, 4, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39, 41, 43, 44, 46, 48, 50, 52, 54, 56, 57, 59, 61, 63, 65, 67, 68, 70, 72, 74, 76, 78, 80, 81, 83, 85, 87, 89, 91, 93, 94, 96, 98, 100, 102, 104, 106, 107, 109, 111, 113, 115, 117, 119, 120, 122, 124, 126 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_4_nl_params_data, conv2d_4_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_4_scratch0, &conv2d_4_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_7_layer, AI_STATIC,
  .tensors = &conv2d_4_chain, 
  .groups = 1, 
  .nl_params = &conv2d_4_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_7_nl_params_data[] = { -128, -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -106, -105, -105, -105, -103, -101, -100, -98, -96, -94, -92, -90, -89, -87, -85, -83, -81, -80, -78, -76, -74, -72, -70, -69, -67, -65, -63, -61, -60, -58, -56, -54, -52, -51, -49, -47, -45, -43, -41, -40, -38, -36, -34, -32, -31, -29, -27, -25, -23, -21, -20, -18, -16, -14, -12, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 20, 22, 24, 26, 28, 29, 31, 33, 35, 37, 39, 40, 42, 44, 46, 48, 49, 51, 53, 55, 57, 58, 60, 62, 64, 66, 68, 69, 71, 73, 75, 77, 78, 80, 82, 84, 86, 88, 89, 91, 93, 95, 97, 98, 100, 102, 104, 106, 108, 109, 111, 113, 115, 117, 118, 120, 122, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_7_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_7_nl_params_data, conv2d_7_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_7_scratch0, &conv2d_7_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_9_layer, AI_STATIC,
  .tensors = &conv2d_7_chain, 
  .groups = 8, 
  .nl_params = &conv2d_7_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_9_weights, &conv2d_9_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_9_layer, 9,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_11_layer, AI_STATIC,
  .tensors = &conv2d_9_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_11_nl_params_data[] = { -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -115, -115, -114, -114, -114, -114, -113, -111, -110, -108, -107, -105, -104, -102, -101, -99, -98, -96, -95, -93, -92, -90, -89, -87, -86, -84, -83, -81, -80, -78, -77, -75, -74, -72, -71, -69, -68, -66, -65, -63, -62, -60, -59, -57, -56, -54, -53, -51, -50, -48, -47, -45, -44, -42, -41, -39, -38, -36, -35, -33, -32, -30, -29, -27, -26, -24, -23, -21, -20, -18, -17, -15, -14, -12, -11, -9, -8, -6, -5, -3, -2, 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 104, 106, 107, 109, 110, 112, 113, 115, 116, 118, 119, 121, 122, 124, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_11_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_11_nl_params_data, conv2d_11_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_11_scratch0, &conv2d_11_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_13_layer, AI_STATIC,
  .tensors = &conv2d_11_chain, 
  .groups = 1, 
  .nl_params = &conv2d_11_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_13_nl_params_data[] = { -128, -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -105, -105, -105, -105, -105, -104, -104, -104, -104, -103, -103, -103, -103, -103, -102, -102, -102, -102, -102, -101, -101, -101, -101, -100, -100, -100, -100, -100, -99, -99, -99, -99, -98, -98, -98, -98, -98, -97, -97, -97, -97, -97, -96, -96, -96, -96, -95, -95, -95, -93, -91, -88, -86, -84, -82, -80, -78, -75, -73, -71, -69, -67, -65, -62, -60, -58, -56, -54, -51, -49, -47, -45, -43, -41, -38, -36, -34, -32, -30, -28, -25, -23, -21, -19, -17, -15, -12, -10, -8, -6, -4, -1, 1, 3, 5, 7, 9, 12, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 46, 49, 51, 53, 55, 57, 59, 62, 64, 66, 68, 70, 73, 75, 77, 79, 81, 83, 86, 88, 90, 92, 94, 96, 99, 101, 103, 105, 107, 109, 112, 114, 116, 118, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_13_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_13_nl_params_data, conv2d_13_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_13_weights, &conv2d_13_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_13_scratch0, &conv2d_13_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_layer, 13,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_15_layer, AI_STATIC,
  .tensors = &conv2d_13_chain, 
  .groups = 1, 
  .nl_params = &conv2d_13_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_15_nl_params_data[] = { -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -104, -104, -104, -104, -103, -103, -103, -103, -103, -102, -102, -102, -102, -102, -101, -101, -101, -101, -101, -100, -100, -100, -100, -100, -99, -99, -99, -99, -98, -98, -98, -98, -98, -97, -97, -97, -95, -93, -91, -89, -87, -84, -82, -80, -78, -76, -74, -72, -70, -68, -66, -64, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -17, -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 117, 119, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_15_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_15_nl_params_data, conv2d_15_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_15_weights, &conv2d_15_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_15_scratch0, &conv2d_15_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_layer, 15,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_17_layer, AI_STATIC,
  .tensors = &conv2d_15_chain, 
  .groups = 48, 
  .nl_params = &conv2d_15_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &concat_20_layer, AI_STATIC,
  .tensors = &conv2d_17_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_17_output, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_20_layer, 20,
  CONCAT_TYPE,
  concat, forward_concat,
  &AI_NET_OBJ_INSTANCE, &conv2d_21_layer, AI_STATIC,
  .tensors = &concat_20_chain, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i8 conv2d_21_nl_params_data[] = { -128, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -106, -105, -105, -105, -103, -101, -100, -98, -96, -94, -92, -91, -89, -87, -85, -83, -82, -80, -78, -76, -74, -73, -71, -69, -67, -65, -64, -62, -60, -58, -56, -55, -53, -51, -49, -47, -46, -44, -42, -40, -38, -37, -35, -33, -31, -29, -28, -26, -24, -22, -20, -19, -17, -15, -13, -12, -10, -8, -6, -4, -3, -1, 1, 3, 5, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 24, 26, 28, 30, 32, 33, 35, 37, 39, 41, 42, 44, 46, 48, 50, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 69, 71, 73, 75, 77, 78, 80, 82, 84, 86, 87, 89, 91, 93, 95, 96, 98, 100, 102, 104, 105, 107, 109, 111, 113, 114, 116, 118, 120, 122, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_21_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_21_nl_params_data, conv2d_21_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_21_weights, &conv2d_21_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_21_scratch0, &conv2d_21_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_21_layer, 21,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_24_layer, AI_STATIC,
  .tensors = &conv2d_21_chain, 
  .groups = 1, 
  .nl_params = &conv2d_21_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_24_nl_params_data[] = { -128, -128, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -104, -104, -104, -104, -104, -103, -103, -103, -103, -103, -102, -102, -102, -102, -101, -101, -101, -101, -101, -100, -100, -100, -100, -100, -99, -99, -99, -99, -98, -98, -98, -98, -98, -97, -97, -97, -97, -97, -96, -96, -96, -96, -95, -95, -95, -93, -91, -89, -86, -84, -82, -80, -78, -76, -73, -71, -69, -67, -65, -63, -61, -58, -56, -54, -52, -50, -48, -45, -43, -41, -39, -37, -35, -33, -30, -28, -26, -24, -22, -20, -17, -15, -13, -11, -9, -7, -5, -2, 0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 26, 28, 30, 32, 34, 36, 38, 41, 43, 45, 47, 49, 51, 54, 56, 58, 60, 62, 64, 66, 69, 71, 73, 75, 77, 79, 82, 84, 86, 88, 90, 92, 94, 97, 99, 101, 103, 105, 107, 110, 112, 114, 116, 118, 120, 122, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_24_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_24_nl_params_data, conv2d_24_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_24_weights, &conv2d_24_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_24_scratch0, &conv2d_24_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_24_layer, 24,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_26_layer, AI_STATIC,
  .tensors = &conv2d_24_chain, 
  .groups = 64, 
  .nl_params = &conv2d_24_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 26,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_27_layer, AI_STATIC,
  .tensors = &conv2d_26_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_27_nl_params_data[] = { -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -104, -104, -104, -104, -104, -103, -103, -103, -103, -102, -102, -102, -102, -102, -101, -101, -101, -101, -101, -100, -100, -100, -100, -100, -99, -99, -99, -99, -99, -98, -98, -98, -96, -94, -92, -90, -88, -86, -84, -82, -80, -78, -76, -74, -71, -69, -67, -65, -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39, -37, -35, -33, -31, -29, -27, -25, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_27_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_27_nl_params_data, conv2d_27_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_27_weights, &conv2d_27_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_27_scratch0, &conv2d_27_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_27_layer, 27,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_29_layer, AI_STATIC,
  .tensors = &conv2d_27_chain, 
  .groups = 1, 
  .nl_params = &conv2d_27_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_29_nl_params_data[] = { -128, -127, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -121, -120, -120, -120, -120, -119, -119, -119, -119, -118, -118, -118, -118, -117, -117, -117, -117, -116, -116, -116, -116, -115, -115, -115, -115, -114, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -112, -111, -111, -111, -111, -110, -110, -110, -110, -109, -109, -109, -109, -108, -108, -108, -108, -107, -107, -107, -107, -106, -106, -106, -106, -105, -105, -105, -105, -104, -104, -104, -104, -103, -103, -103, -103, -102, -102, -102, -102, -101, -101, -101, -101, -100, -100, -100, -100, -99, -99, -99, -99, -98, -98, -98, -98, -97, -97, -97, -97, -96, -96, -96, -96, -95, -95, -95, -95, -94, -94, -94, -94, -93, -93, -93, -93, -92, -92, -92, -92, -91, -91, -91, -91, -90, -90, -90, -90, -89, -89, -89, -89, -88, -88, -88, -88, -87, -87, -87, -87, -86, -86, -86, -86, -85, -85, -82, -80, -77, -75, -72, -70, -67, -65, -62, -60, -57, -55, -52, -50, -47, -45, -42, -40, -37, -35, -32, -30, -27, -25, -22, -20, -17, -15, -12, -10, -7, -5, -2, 0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55, 58, 60, 63, 65, 68, 70, 73, 75, 78, 80, 83, 85, 88, 90, 93, 95, 98, 100, 103, 105, 108, 110, 113, 115, 118, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_29_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_29_nl_params_data, conv2d_29_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_29_weights, &conv2d_29_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_29_scratch0, &conv2d_29_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_29_layer, 29,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_31_layer, AI_STATIC,
  .tensors = &conv2d_29_chain, 
  .groups = 96, 
  .nl_params = &conv2d_29_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_29_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_31_weights, &conv2d_31_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_31_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_31_layer, 31,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &eltwise_32_layer, AI_STATIC,
  .tensors = &conv2d_31_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_26_output, &conv2d_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_32_layer, 32,
  ELTWISE_TYPE,
  eltwise, forward_add_integer_INT8,
  &AI_NET_OBJ_INSTANCE, &conv2d_33_layer, AI_STATIC,
  .tensors = &eltwise_32_chain, 
)


AI_STATIC_CONST ai_i8 conv2d_33_nl_params_data[] = { -128, -128, -128, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -105, -104, -104, -104, -104, -104, -103, -103, -103, -101, -99, -97, -95, -94, -92, -90, -88, -86, -84, -82, -80, -79, -77, -75, -73, -71, -69, -67, -65, -63, -62, -60, -58, -56, -54, -52, -50, -48, -46, -45, -43, -41, -39, -37, -35, -33, -31, -30, -28, -26, -24, -22, -20, -18, -16, -14, -13, -11, -9, -7, -5, -3, -1, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 21, 23, 25, 27, 29, 31, 33, 35, 36, 38, 40, 42, 44, 46, 48, 50, 51, 53, 55, 57, 59, 61, 63, 65, 67, 68, 70, 72, 74, 76, 78, 80, 82, 83, 85, 87, 89, 91, 93, 95, 97, 99, 100, 102, 104, 106, 108, 110, 112, 114, 115, 117, 119, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_33_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_33_nl_params_data, conv2d_33_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_33_weights, &conv2d_33_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_33_scratch0, &conv2d_33_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_33_layer, 33,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_35_layer, AI_STATIC,
  .tensors = &conv2d_33_chain, 
  .groups = 1, 
  .nl_params = &conv2d_33_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_35_nl_params_data[] = { -128, -128, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -105, -105, -105, -105, -105, -104, -104, -104, -104, -103, -103, -103, -103, -103, -102, -102, -102, -102, -101, -101, -101, -101, -101, -100, -100, -100, -100, -99, -99, -99, -99, -99, -98, -98, -98, -98, -98, -97, -97, -97, -97, -96, -96, -96, -96, -96, -95, -95, -95, -95, -94, -94, -94, -92, -90, -87, -85, -83, -81, -79, -76, -74, -72, -70, -68, -65, -63, -61, -59, -57, -54, -52, -50, -48, -46, -44, -41, -39, -37, -35, -33, -30, -28, -26, -24, -22, -19, -17, -15, -13, -11, -8, -6, -4, -2, 0, 3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 46, 49, 51, 53, 55, 57, 60, 62, 64, 66, 68, 71, 73, 75, 77, 79, 82, 84, 86, 88, 90, 93, 95, 97, 99, 101, 104, 106, 108, 110, 112, 115, 117, 119, 121, 123, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_35_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_35_nl_params_data, conv2d_35_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_35_weights, &conv2d_35_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_35_scratch0, &conv2d_35_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_35_layer, 35,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_37_layer, AI_STATIC,
  .tensors = &conv2d_35_chain, 
  .groups = 96, 
  .nl_params = &conv2d_35_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_37_weights, &conv2d_37_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_37_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_37_layer, 37,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &eltwise_38_layer, AI_STATIC,
  .tensors = &conv2d_37_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_32_output, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_38_layer, 38,
  ELTWISE_TYPE,
  eltwise, forward_add_integer_INT8,
  &AI_NET_OBJ_INSTANCE, &conv2d_39_layer, AI_STATIC,
  .tensors = &eltwise_38_chain, 
)


AI_STATIC_CONST ai_i8 conv2d_39_nl_params_data[] = { -128, -128, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -107, -106, -106, -106, -104, -102, -101, -99, -97, -95, -94, -92, -90, -88, -87, -85, -83, -81, -79, -78, -76, -74, -72, -71, -69, -67, -65, -64, -62, -60, -58, -57, -55, -53, -51, -49, -48, -46, -44, -42, -41, -39, -37, -35, -34, -32, -30, -28, -26, -25, -23, -21, -19, -18, -16, -14, -12, -11, -9, -7, -5, -3, -2, 0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 25, 27, 28, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 48, 50, 51, 53, 55, 57, 58, 60, 62, 64, 65, 67, 69, 71, 73, 74, 76, 78, 80, 81, 83, 85, 87, 88, 90, 92, 94, 95, 97, 99, 101, 103, 104, 106, 108, 110, 111, 113, 115, 117, 118, 120, 122, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_39_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_39_nl_params_data, conv2d_39_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_39_weights, &conv2d_39_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_39_scratch0, &conv2d_39_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_39_layer, 39,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_41_layer, AI_STATIC,
  .tensors = &conv2d_39_chain, 
  .groups = 1, 
  .nl_params = &conv2d_39_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_41_nl_params_data[] = { -128, -128, -127, -127, -127, -126, -126, -126, -126, -125, -125, -125, -124, -124, -124, -124, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -120, -120, -120, -120, -119, -119, -119, -118, -118, -118, -118, -117, -117, -117, -117, -116, -116, -116, -115, -115, -115, -115, -114, -114, -114, -113, -113, -113, -113, -112, -112, -112, -111, -111, -111, -111, -110, -110, -110, -109, -109, -109, -109, -108, -108, -108, -107, -107, -107, -107, -106, -106, -106, -105, -105, -105, -105, -104, -104, -104, -103, -103, -103, -103, -102, -102, -102, -101, -101, -101, -101, -100, -100, -100, -100, -99, -99, -99, -98, -98, -98, -98, -97, -97, -97, -96, -96, -96, -96, -95, -95, -95, -94, -94, -94, -94, -93, -93, -93, -92, -92, -92, -92, -91, -91, -91, -90, -90, -90, -90, -89, -89, -89, -88, -88, -88, -88, -87, -87, -87, -86, -86, -86, -86, -85, -85, -85, -84, -84, -84, -84, -83, -83, -83, -83, -82, -82, -82, -81, -81, -81, -81, -80, -80, -80, -79, -79, -79, -79, -78, -78, -78, -77, -77, -77, -77, -76, -76, -73, -70, -68, -65, -62, -59, -56, -53, -51, -48, -45, -42, -39, -36, -34, -31, -28, -25, -22, -19, -17, -14, -11, -8, -5, -2, 0, 3, 6, 9, 12, 15, 17, 20, 23, 26, 29, 32, 34, 37, 40, 43, 46, 49, 51, 54, 57, 60, 63, 66, 68, 71, 74, 77, 80, 83, 85, 88, 91, 94, 97, 100, 102, 105, 108, 111, 114, 117, 119, 122, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_41_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_41_nl_params_data, conv2d_41_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_41_weights, &conv2d_41_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_41_scratch0, &conv2d_41_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_41_layer, 41,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_43_layer, AI_STATIC,
  .tensors = &conv2d_41_chain, 
  .groups = 96, 
  .nl_params = &conv2d_41_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_43_weights, &conv2d_43_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_43_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_43_layer, 43,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &eltwise_44_layer, AI_STATIC,
  .tensors = &conv2d_43_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_38_output, &conv2d_43_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_44_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_44_layer, 44,
  ELTWISE_TYPE,
  eltwise, forward_add_integer_INT8,
  &AI_NET_OBJ_INSTANCE, &concat_47_layer, AI_STATIC,
  .tensors = &eltwise_44_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_44_output, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_47_layer, 47,
  CONCAT_TYPE,
  concat, forward_concat,
  &AI_NET_OBJ_INSTANCE, &conv2d_48_layer, AI_STATIC,
  .tensors = &concat_47_chain, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_i8 conv2d_48_nl_params_data[] = { -128, -128, -128, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -108, -107, -107, -107, -105, -104, -102, -100, -98, -97, -95, -93, -91, -90, -88, -86, -84, -83, -81, -79, -77, -76, -74, -72, -70, -69, -67, -65, -64, -62, -60, -58, -57, -55, -53, -51, -50, -48, -46, -44, -43, -41, -39, -37, -36, -34, -32, -30, -29, -27, -25, -24, -22, -20, -18, -17, -15, -13, -11, -10, -8, -6, -4, -3, -1, 1, 3, 4, 6, 8, 10, 11, 13, 15, 17, 18, 20, 22, 23, 25, 27, 29, 30, 32, 34, 36, 37, 39, 41, 43, 44, 46, 48, 50, 51, 53, 55, 57, 58, 60, 62, 63, 65, 67, 69, 70, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 90, 91, 93, 95, 97, 98, 100, 102, 103, 105, 107, 109, 110, 112, 114, 116, 117, 119, 121, 123, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_48_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_48_nl_params_data, conv2d_48_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_47_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_48_weights, &conv2d_48_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_48_scratch0, &conv2d_48_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_48_layer, 48,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_51_layer, AI_STATIC,
  .tensors = &conv2d_48_chain, 
  .groups = 1, 
  .nl_params = &conv2d_48_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_i8 conv2d_51_nl_params_data[] = { -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -113, -113, -113, -113, -113, -112, -112, -112, -112, -112, -112, -111, -111, -111, -111, -111, -110, -110, -110, -110, -110, -109, -109, -109, -109, -109, -109, -108, -108, -108, -108, -108, -107, -107, -107, -107, -107, -106, -106, -106, -106, -106, -105, -105, -105, -105, -105, -105, -104, -104, -104, -104, -104, -103, -103, -103, -101, -99, -97, -95, -94, -92, -90, -88, -86, -84, -82, -80, -78, -76, -75, -73, -71, -69, -67, -65, -63, -61, -59, -57, -56, -54, -52, -50, -48, -46, -44, -42, -40, -38, -37, -35, -33, -31, -29, -27, -25, -23, -21, -19, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 96, 98, 100, 102, 104, 106, 108, 110, 112, 113, 115, 117, 119, 121, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_51_nl_params, AI_ARRAY_FORMAT_S8,
    conv2d_51_nl_params_data, conv2d_51_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_51_weights, &conv2d_51_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_51_scratch0, &conv2d_51_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_51_layer, 51,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_53_layer, AI_STATIC,
  .tensors = &conv2d_51_chain, 
  .groups = 128, 
  .nl_params = &conv2d_51_nl_params, 
  .nl_func = nl_func_array_integer, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_51_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_53_weights, &conv2d_53_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_53_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_53_layer, 53,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_54_layer, AI_STATIC,
  .tensors = &conv2d_53_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_54_weights, &conv2d_54_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_54_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_54_layer, 54,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conv2d_55_layer, AI_STATIC,
  .tensors = &conv2d_54_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_55_weights, &conv2d_55_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_55_layer, 55,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer_SSSA_ch,
  &AI_NET_OBJ_INSTANCE, &conversion_56_layer, AI_STATIC,
  .tensors = &conv2d_55_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_55_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_56_layer, 56,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &conversion_56_layer, AI_STATIC,
  .tensors = &conversion_56_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 73064, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 53116, 1,
                     NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PERSON_YOLO_IN_NUM, &image_input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PERSON_YOLO_OUT_NUM, &conversion_56_output),
  &conversion_0_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool person_yolo_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, AI_PERSON_YOLO_ACTIVATIONS_ALIGNMENT));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2d_55_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_55_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_54_scratch0_array.data = AI_PTR(activations + 2752);
    conv2d_54_scratch0_array.data_start = AI_PTR(activations + 2752);
    conv2d_53_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_53_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_51_scratch1_array.data = AI_PTR(activations + 4740);
    conv2d_51_scratch1_array.data_start = AI_PTR(activations + 4740);
    conv2d_51_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_51_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_48_scratch1_array.data = AI_PTR(activations + 1632);
    conv2d_48_scratch1_array.data_start = AI_PTR(activations + 1632);
    conv2d_48_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_48_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_43_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_43_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_41_scratch1_array.data = AI_PTR(activations + 34372);
    conv2d_41_scratch1_array.data_start = AI_PTR(activations + 34372);
    conv2d_41_scratch0_array.data = AI_PTR(activations + 15172);
    conv2d_41_scratch0_array.data_start = AI_PTR(activations + 15172);
    conv2d_39_scratch1_array.data = AI_PTR(activations + 15172);
    conv2d_39_scratch1_array.data_start = AI_PTR(activations + 15172);
    conv2d_39_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_39_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_37_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_37_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_35_scratch1_array.data = AI_PTR(activations + 15172);
    conv2d_35_scratch1_array.data_start = AI_PTR(activations + 15172);
    conv2d_35_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_35_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_33_scratch1_array.data = AI_PTR(activations + 15172);
    conv2d_33_scratch1_array.data_start = AI_PTR(activations + 15172);
    conv2d_33_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_33_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_31_scratch0_array.data = AI_PTR(activations + 2896);
    conv2d_31_scratch0_array.data_start = AI_PTR(activations + 2896);
    conv2d_29_scratch1_array.data = AI_PTR(activations + 15172);
    conv2d_29_scratch1_array.data_start = AI_PTR(activations + 15172);
    conv2d_29_scratch0_array.data = AI_PTR(activations + 2896);
    conv2d_29_scratch0_array.data_start = AI_PTR(activations + 2896);
    conv2d_27_scratch1_array.data = AI_PTR(activations + 15172);
    conv2d_27_scratch1_array.data_start = AI_PTR(activations + 15172);
    conv2d_27_scratch0_array.data = AI_PTR(activations + 2896);
    conv2d_27_scratch0_array.data_start = AI_PTR(activations + 2896);
    conv2d_26_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_26_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_24_scratch1_array.data = AI_PTR(activations + 2372);
    conv2d_24_scratch1_array.data_start = AI_PTR(activations + 2372);
    conv2d_24_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_24_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_21_scratch1_array.data = AI_PTR(activations + 16832);
    conv2d_21_scratch1_array.data_start = AI_PTR(activations + 16832);
    conv2d_21_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_21_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_17_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_17_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_15_scratch1_array.data = AI_PTR(activations + 7232);
    conv2d_15_scratch1_array.data_start = AI_PTR(activations + 7232);
    conv2d_15_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_15_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_13_scratch1_array.data = AI_PTR(activations + 7232);
    conv2d_13_scratch1_array.data_start = AI_PTR(activations + 7232);
    conv2d_13_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_13_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_11_scratch1_array.data = AI_PTR(activations + 832);
    conv2d_11_scratch1_array.data_start = AI_PTR(activations + 832);
    conv2d_11_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_11_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_9_scratch0_array.data = AI_PTR(activations + 12800);
    conv2d_9_scratch0_array.data_start = AI_PTR(activations + 12800);
    conv2d_7_scratch1_array.data = AI_PTR(activations + 0);
    conv2d_7_scratch1_array.data_start = AI_PTR(activations + 0);
    conv2d_7_scratch0_array.data = AI_PTR(activations + 51528);
    conv2d_7_scratch0_array.data_start = AI_PTR(activations + 51528);
    conv2d_4_scratch1_array.data = AI_PTR(activations + 328);
    conv2d_4_scratch1_array.data_start = AI_PTR(activations + 328);
    conv2d_4_scratch0_array.data = AI_PTR(activations + 52176);
    conv2d_4_scratch0_array.data_start = AI_PTR(activations + 52176);
    conv2d_2_scratch1_array.data = AI_PTR(activations + 976);
    conv2d_2_scratch1_array.data_start = AI_PTR(activations + 976);
    conv2d_2_scratch0_array.data = AI_PTR(activations + 52824);
    conv2d_2_scratch0_array.data_start = AI_PTR(activations + 52824);
    image_input_output_array.data = AI_PTR(NULL);
    image_input_output_array.data_start = AI_PTR(NULL);
    conversion_0_output_array.data = AI_PTR(activations + 27224);
    conversion_0_output_array.data_start = AI_PTR(activations + 27224);
    conv2d_2_output_array.data = AI_PTR(activations + 976);
    conv2d_2_output_array.data_start = AI_PTR(activations + 976);
    conv2d_4_output_array.data = AI_PTR(activations + 328);
    conv2d_4_output_array.data_start = AI_PTR(activations + 328);
    conv2d_7_output_array.data = AI_PTR(activations + 0);
    conv2d_7_output_array.data_start = AI_PTR(activations + 0);
    conv2d_9_output_array.data = AI_PTR(activations + 12872);
    conv2d_9_output_array.data_start = AI_PTR(activations + 12872);
    conv2d_11_output_array.data = AI_PTR(activations + 4032);
    conv2d_11_output_array.data_start = AI_PTR(activations + 4032);
    conv2d_13_output_array.data = AI_PTR(activations + 26432);
    conv2d_13_output_array.data_start = AI_PTR(activations + 26432);
    conv2d_15_output_array.data = AI_PTR(activations + 7232);
    conv2d_15_output_array.data_start = AI_PTR(activations + 7232);
    conv2d_17_output_array.data = AI_PTR(activations + 26432);
    conv2d_17_output_array.data_start = AI_PTR(activations + 26432);
    concat_20_output_array.data = AI_PTR(activations + 7232);
    concat_20_output_array.data_start = AI_PTR(activations + 7232);
    conv2d_21_output_array.data = AI_PTR(activations + 16832);
    conv2d_21_output_array.data_start = AI_PTR(activations + 16832);
    conv2d_24_output_array.data = AI_PTR(activations + 8772);
    conv2d_24_output_array.data_start = AI_PTR(activations + 8772);
    conv2d_26_output_array.data = AI_PTR(activations + 496);
    conv2d_26_output_array.data_start = AI_PTR(activations + 496);
    conv2d_27_output_array.data = AI_PTR(activations + 24772);
    conv2d_27_output_array.data_start = AI_PTR(activations + 24772);
    conv2d_29_output_array.data = AI_PTR(activations + 34372);
    conv2d_29_output_array.data_start = AI_PTR(activations + 34372);
    conv2d_31_output_array.data = AI_PTR(activations + 3520);
    conv2d_31_output_array.data_start = AI_PTR(activations + 3520);
    eltwise_32_output_array.data = AI_PTR(activations + 5920);
    eltwise_32_output_array.data_start = AI_PTR(activations + 5920);
    conv2d_33_output_array.data = AI_PTR(activations + 24772);
    conv2d_33_output_array.data_start = AI_PTR(activations + 24772);
    conv2d_35_output_array.data = AI_PTR(activations + 34372);
    conv2d_35_output_array.data_start = AI_PTR(activations + 34372);
    conv2d_37_output_array.data = AI_PTR(activations + 624);
    conv2d_37_output_array.data_start = AI_PTR(activations + 624);
    eltwise_38_output_array.data = AI_PTR(activations + 3024);
    eltwise_38_output_array.data_start = AI_PTR(activations + 3024);
    conv2d_39_output_array.data = AI_PTR(activations + 24772);
    conv2d_39_output_array.data_start = AI_PTR(activations + 24772);
    conv2d_41_output_array.data = AI_PTR(activations + 34372);
    conv2d_41_output_array.data_start = AI_PTR(activations + 34372);
    conv2d_43_output_array.data = AI_PTR(activations + 624);
    conv2d_43_output_array.data_start = AI_PTR(activations + 624);
    eltwise_44_output_array.data = AI_PTR(activations + 5424);
    eltwise_44_output_array.data_start = AI_PTR(activations + 5424);
    concat_47_output_array.data = AI_PTR(activations + 15172);
    concat_47_output_array.data_start = AI_PTR(activations + 15172);
    conv2d_48_output_array.data = AI_PTR(activations + 23972);
    conv2d_48_output_array.data_start = AI_PTR(activations + 23972);
    conv2d_51_output_array.data = AI_PTR(activations + 7940);
    conv2d_51_output_array.data_start = AI_PTR(activations + 7940);
    conv2d_53_output_array.data = AI_PTR(activations + 1152);
    conv2d_53_output_array.data_start = AI_PTR(activations + 1152);
    conv2d_54_output_array.data = AI_PTR(activations + 5568);
    conv2d_54_output_array.data_start = AI_PTR(activations + 5568);
    conv2d_55_output_array.data = AI_PTR(activations + 1324);
    conv2d_55_output_array.data_start = AI_PTR(activations + 1324);
    conversion_56_output_array.data = AI_PTR(NULL);
    conversion_56_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool person_yolo_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    conv2d_55_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_55_bias_array.data = AI_PTR(weights + 72944);
    conv2d_55_bias_array.data_start = AI_PTR(weights + 72944);
    conv2d_55_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_55_weights_array.data = AI_PTR(weights + 65264);
    conv2d_55_weights_array.data_start = AI_PTR(weights + 65264);
    conv2d_54_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_54_bias_array.data = AI_PTR(weights + 64240);
    conv2d_54_bias_array.data_start = AI_PTR(weights + 64240);
    conv2d_54_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_54_weights_array.data = AI_PTR(weights + 47856);
    conv2d_54_weights_array.data_start = AI_PTR(weights + 47856);
    conv2d_53_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_53_bias_array.data = AI_PTR(weights + 47600);
    conv2d_53_bias_array.data_start = AI_PTR(weights + 47600);
    conv2d_53_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_53_weights_array.data = AI_PTR(weights + 39408);
    conv2d_53_weights_array.data_start = AI_PTR(weights + 39408);
    conv2d_51_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_51_bias_array.data = AI_PTR(weights + 38896);
    conv2d_51_bias_array.data_start = AI_PTR(weights + 38896);
    conv2d_51_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_51_weights_array.data = AI_PTR(weights + 37744);
    conv2d_51_weights_array.data_start = AI_PTR(weights + 37744);
    conv2d_48_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_48_bias_array.data = AI_PTR(weights + 37232);
    conv2d_48_bias_array.data_start = AI_PTR(weights + 37232);
    conv2d_48_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_48_weights_array.data = AI_PTR(weights + 25968);
    conv2d_48_weights_array.data_start = AI_PTR(weights + 25968);
    conv2d_43_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_43_bias_array.data = AI_PTR(weights + 25872);
    conv2d_43_bias_array.data_start = AI_PTR(weights + 25872);
    conv2d_43_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_43_weights_array.data = AI_PTR(weights + 23568);
    conv2d_43_weights_array.data_start = AI_PTR(weights + 23568);
    conv2d_41_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_bias_array.data = AI_PTR(weights + 23184);
    conv2d_41_bias_array.data_start = AI_PTR(weights + 23184);
    conv2d_41_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_41_weights_array.data = AI_PTR(weights + 22320);
    conv2d_41_weights_array.data_start = AI_PTR(weights + 22320);
    conv2d_39_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_bias_array.data = AI_PTR(weights + 21936);
    conv2d_39_bias_array.data_start = AI_PTR(weights + 21936);
    conv2d_39_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_39_weights_array.data = AI_PTR(weights + 19632);
    conv2d_39_weights_array.data_start = AI_PTR(weights + 19632);
    conv2d_37_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_bias_array.data = AI_PTR(weights + 19536);
    conv2d_37_bias_array.data_start = AI_PTR(weights + 19536);
    conv2d_37_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_37_weights_array.data = AI_PTR(weights + 17232);
    conv2d_37_weights_array.data_start = AI_PTR(weights + 17232);
    conv2d_35_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_35_bias_array.data = AI_PTR(weights + 16848);
    conv2d_35_bias_array.data_start = AI_PTR(weights + 16848);
    conv2d_35_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_35_weights_array.data = AI_PTR(weights + 15984);
    conv2d_35_weights_array.data_start = AI_PTR(weights + 15984);
    conv2d_33_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_bias_array.data = AI_PTR(weights + 15600);
    conv2d_33_bias_array.data_start = AI_PTR(weights + 15600);
    conv2d_33_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_33_weights_array.data = AI_PTR(weights + 13296);
    conv2d_33_weights_array.data_start = AI_PTR(weights + 13296);
    conv2d_31_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_31_bias_array.data = AI_PTR(weights + 13200);
    conv2d_31_bias_array.data_start = AI_PTR(weights + 13200);
    conv2d_31_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_31_weights_array.data = AI_PTR(weights + 10896);
    conv2d_31_weights_array.data_start = AI_PTR(weights + 10896);
    conv2d_29_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_bias_array.data = AI_PTR(weights + 10512);
    conv2d_29_bias_array.data_start = AI_PTR(weights + 10512);
    conv2d_29_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_29_weights_array.data = AI_PTR(weights + 9648);
    conv2d_29_weights_array.data_start = AI_PTR(weights + 9648);
    conv2d_27_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_27_bias_array.data = AI_PTR(weights + 9264);
    conv2d_27_bias_array.data_start = AI_PTR(weights + 9264);
    conv2d_27_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_27_weights_array.data = AI_PTR(weights + 6960);
    conv2d_27_weights_array.data_start = AI_PTR(weights + 6960);
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(weights + 6864);
    conv2d_26_bias_array.data_start = AI_PTR(weights + 6864);
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(weights + 5328);
    conv2d_26_weights_array.data_start = AI_PTR(weights + 5328);
    conv2d_24_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_bias_array.data = AI_PTR(weights + 5072);
    conv2d_24_bias_array.data_start = AI_PTR(weights + 5072);
    conv2d_24_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_weights_array.data = AI_PTR(weights + 4496);
    conv2d_24_weights_array.data_start = AI_PTR(weights + 4496);
    conv2d_21_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_bias_array.data = AI_PTR(weights + 4240);
    conv2d_21_bias_array.data_start = AI_PTR(weights + 4240);
    conv2d_21_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_weights_array.data = AI_PTR(weights + 2704);
    conv2d_21_weights_array.data_start = AI_PTR(weights + 2704);
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(weights + 2640);
    conv2d_17_bias_array.data_start = AI_PTR(weights + 2640);
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(weights + 1872);
    conv2d_17_weights_array.data_start = AI_PTR(weights + 1872);
    conv2d_15_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_bias_array.data = AI_PTR(weights + 1680);
    conv2d_15_bias_array.data_start = AI_PTR(weights + 1680);
    conv2d_15_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_weights_array.data = AI_PTR(weights + 1248);
    conv2d_15_weights_array.data_start = AI_PTR(weights + 1248);
    conv2d_13_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_bias_array.data = AI_PTR(weights + 1056);
    conv2d_13_bias_array.data_start = AI_PTR(weights + 1056);
    conv2d_13_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_weights_array.data = AI_PTR(weights + 672);
    conv2d_13_weights_array.data_start = AI_PTR(weights + 672);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(weights + 640);
    conv2d_11_bias_array.data_start = AI_PTR(weights + 640);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(weights + 352);
    conv2d_11_weights_array.data_start = AI_PTR(weights + 352);
    conv2d_9_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_bias_array.data = AI_PTR(weights + 336);
    conv2d_9_bias_array.data_start = AI_PTR(weights + 336);
    conv2d_9_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_weights_array.data = AI_PTR(weights + 304);
    conv2d_9_weights_array.data_start = AI_PTR(weights + 304);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(weights + 272);
    conv2d_7_bias_array.data_start = AI_PTR(weights + 272);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(weights + 200);
    conv2d_7_weights_array.data_start = AI_PTR(weights + 200);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(weights + 168);
    conv2d_4_bias_array.data_start = AI_PTR(weights + 168);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(weights + 104);
    conv2d_4_weights_array.data_start = AI_PTR(weights + 104);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(weights + 72);
    conv2d_2_bias_array.data_start = AI_PTR(weights + 72);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(weights + 0);
    conv2d_2_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_person_yolo_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_PERSON_YOLO_MODEL_NAME,
      .model_signature   = AI_PERSON_YOLO_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6538254,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_person_yolo_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_person_yolo_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_person_yolo_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_person_yolo_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= person_yolo_configure_weights(net_ctx, &params->params);
  ok &= person_yolo_configure_activations(net_ctx, &params->activations);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_person_yolo_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_person_yolo_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}




#undef AI_PERSON_YOLO_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

