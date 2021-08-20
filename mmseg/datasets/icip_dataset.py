import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

from PIL import Image
import numpy as np

@DATASETS.register_module()
class ICIPRGBDataset(CustomDataset):
    """ICIP dataset
    """

    CLASSES = ('background', 'trans', 'mirror')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

    def __init__(self, test_outpath=None, **kwargs):
        super(ICIPRGBDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
        self.test_outpath = test_outpath

    def format_results(self, results, **kwargs):
        for meta, mask in zip(self.img_infos, results):
            im = Image.fromarray(np.uint8(mask))
            # TODO use kwargs
            im.save(osp.join(self.test_outpath, meta['filename'].rstrip('.jpg'))+'.png')

@DATASETS.register_module()
class ICIPRGBDDataset(CustomDataset):
    """ICIP dataset
    """

    CLASSES = ('background', 'trans', 'mirror')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

    def __init__(self, depth_dir, depth_suffix, test_outpath=None, **kwargs):
        super(ICIPRGBDDataset, self).__init__(
                img_suffix='.jpg',
                seg_map_suffix='.png',
                reduce_zero_label=False,
                **kwargs)
        self.depth_dir = osp.join(self.data_root, depth_dir)
        self.depth_suffix = depth_suffix
        self.test_outpath = test_outpath
        assert osp.exists(self.img_dir)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['depth_prefix'] = self.depth_dir
        results['img_suffix'] = self.img_suffix
        results['depth_suffix'] = self.depth_suffix
        if self.custom_classes:
            results['label_map'] = self.label_map

    def format_results(self, results, **kwargs):
        for meta, mask in zip(self.img_infos, results):
            im = Image.fromarray(np.uint8(mask))
            # TODO use kwargs
            im.save(osp.join(self.test_outpath, meta['filename'].rstrip('.jpg'))+'.png')