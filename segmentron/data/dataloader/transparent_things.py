"""Pascal Transparent Semantic Segmentation Dataset."""
import os
import logging
import torch
import numpy as np

from PIL import Image
from .seg_data_base import SegmentationDataset


class TransparentThingsSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = TransparentThingsSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'Trans10K_cls12'
    NUM_CLASS = 12

    def __init__(self, root='datasets/transparent', split='test', mode=None, transform=None, **kwargs):
        super(TransparentThingsSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), f"Please put the data in {root}"
        self.images, self.masks = _get_trans10k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info('Found {} images in the folder {}'.format(len(self.images), root))
        # cup, bottle, jar, bowl and eyeglass
        self.things = [7, 10, 2, 9, 6]
        # windows, shelf, box, freezer, glass walls and glass doors
        self.stuff = [4, 1, 11, 3, 8, 5]
        self.label_map = {0: 0, 2: 1, 6: 2, 7: 3, 9: 4, 10: 5}


    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BILINEAR)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index]).convert("P")
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        # this line sets off any stuff pixels as background as they are not important for tracking purposes
        mask[np.isin(mask, self.stuff)] = 0
        for k in self.label_map.keys():
            mask[mask == k] = self.label_map[k]
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        # only [7, 10, 2, 9, 6] remains 2,6,7,9,10
        #bg shelf Jar freezer window door eyeglass cup wall bowl bottle box
        # 0  1    2      3      4     5       6     7   8     9    10    11
        return ('Background', 'Jar or Tank', 'Eyeglass', 'Cup', 'Glass Bowl', 'Water Bottle')
        #             0             1            2         3         4             5
        # 0=>0 , 2=>1, 6=>2, 7=>3, 9=>4, 10=>5

def _get_trans10k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'train/images')
        mask_folder = os.path.join(folder, 'train/masks_12')
        things = os.path.join(folder, 'things_train.txt')
    elif mode == "val":
        img_folder = os.path.join(folder, 'validation/images')
        mask_folder = os.path.join(folder, 'validation/masks_12')
        things = os.path.join(folder, 'things_validation.txt')
    else:
        assert  mode == "test"
        img_folder = os.path.join(folder, 'test/images')
        mask_folder = os.path.join(folder, 'test/masks_12')
        things = os.path.join(folder, 'things_test.txt')

    things = list(np.genfromtxt(things, delimiter=',', dtype=np.str))

    for filename in os.listdir(img_folder):
        if not filename in things:
            continue
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '_mask.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask:', maskpath)

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = TransparentThingsSegmentation()
