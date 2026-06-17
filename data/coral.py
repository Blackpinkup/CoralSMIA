import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
import ever as er
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from ever.api.data import distributed
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Coral=(255, 0, 0),
)


LABEL_MAP = OrderedDict(
    Background=0,
    Coral=1,
)



def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls



class CoralDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.png'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.jpg'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname.replace('.jpg', '.png')))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        mask=None
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
            if self.transforms is not None:
                if min(mask.shape[:2]) < 512 or min(image.shape[:2]) < 512:
                    print(self.rgb_filepath_list[idx])
                    print(image.shape)
                    print(mask.shape)
                    exit()
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']
            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))
        

    def __len__(self):
        return len(self.rgb_filepath_list)


@er.registry.DATALOADER.register()
class CoralLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = CoralDataset(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
            dataset)

        super(CoralLoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True)
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
