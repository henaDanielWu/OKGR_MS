from functools import partial

import mindspore

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from kitti_dataset import KittiDataset
from .waymo.waymo_dataset import WaymoDataset
import x2ms_adapter
import x2ms_adapter.torch_api.datasets as x2ms_datasets

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'WaymoDataset': WaymoDataset,
}


class DistributedSampler(x2ms_datasets.DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = x2ms_adapter.Generator()
            g.manual_seed(self.epoch)
            indices = x2ms_adapter.tensor_api.tolist(x2ms_adapter.randperm(len(self.dataset), generator=g))
        else:
            indices = x2ms_adapter.tensor_api.tolist(x2ms_adapter.arange(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = x2ms_datasets.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    dataloader=mindspore.dataset.GeneratorDataset(source=dataset,column_names=['data'],shuffle=(sampler is None) and training, sampler=sampler,num_parallel_workers=workers)
    dataloader=dataloader.batch(batch_size,per_batch_map=dataset.collate_batch,drop_remainder=False)
    return dataset, dataloader, sampler
