from collections import namedtuple

import numpy as np

from .detectors import build_detector
import mindspore
import x2ms_adapter

# try:
import kornia
# except:
#     pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict:
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.x2ms_float(kornia.image_to_tensor(val)))
        elif key in ['image_shape']:
            batch_dict[key] = x2ms_adapter.tensor_api.x2ms_int(x2ms_adapter.from_numpy(val))
        else:
            batch_dict[key] = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(val))




# def model_func(model, batch_dict):
#     load_data_to_gpu(batch_dict)
#     ret_dict, tb_dict, disp_dict = model(batch_dict)
#
#     loss = x2ms_adapter.tensor_api.x2ms_mean(ret_dict['loss'])
#     if hasattr(model, 'update_global_step'):
#         model.update_global_step()
#     else:
#         model.module.update_global_step()
#
#     return loss, tb_dict, disp_dict


