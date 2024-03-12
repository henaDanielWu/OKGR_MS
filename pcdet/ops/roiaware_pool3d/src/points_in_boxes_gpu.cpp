//
// Created by 朝阳 on 2023/12/26.
//
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>



extern "C" int CustomFunc(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra){

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}
