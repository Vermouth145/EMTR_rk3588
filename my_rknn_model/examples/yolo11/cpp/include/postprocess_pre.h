#ifndef _RKNN_YOLO11_DEMO_POSTPROCESS_H_
#define _RKNN_YOLO11_DEMO_POSTPROCESS_H_

#include <opencv2/opencv.hpp>

#include "rknn_utils.h"
#include "resize_function.h"
#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 3
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)
//#define NMS_THRESHOLD 0.45
//#define CONF_THRESHOLD 0.25

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
//    BOX_RECT box;
//    float prop;
    int class_index;
    BOX_RECT box;
    float prop;
    int cameraid;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(rknn_context ctx, void *outputs, LETTER_BOX *letter_box, float conf_threshold, float nms_threshold, detect_result_group_t *od_results,int width, int height, rknn_input_output_num io_num, rknn_tensor_attr *output_attrs, bool is_quant, int batch_size);

const char *get_label_name(int class_id);

void deinitPostProcess();
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_