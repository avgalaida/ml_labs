import os
from PIL import Image
from torch.utils.data import Dataset
from utils.mask_transform import rgb_to_multiclass
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class InMemoryDS(Dataset):
    def __init__(self, images_dir, masks_dir, class_rgb_dict):
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.class_rgb = class_rgb_dict

        self.img_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

        self.imgs = []
        self.masks = []
        self.mulclass = []

        self.pre_calc()

    def pre_calc(self):
        transform = Compose([Resize((256, 256)), ToTensor()])

        for i in range(self.__len__()):
            self.imgs.append(transform(Image.open(self.images_dir + self.img_filenames[i])))
            self.masks.append(transform(Image.open(self.masks_dir + self.mask_filenames[i])))
            self.mulclass.append(rgb_to_multiclass(self.masks[-1], self.class_rgb))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        return self.imgs[idx], self.masks[idx], self.mulclass[idx]
