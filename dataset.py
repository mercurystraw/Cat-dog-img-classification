# datasets.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    # 遍历每个类别文件夹（cat 和 dog）对于每个图片文件，构建一个元组，包含图片的完整路径和该图片所属类别的索引（cat 对应 0，dog 对应 1）。
    # 这些元组被存储在 self.imgs 列表中。
    def __init__(self, root, base_transform=None):
        self.root = root
        self.base_transform = base_transform
        self.classes = ['cat', 'dog']
        self.imgs = []
        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    self.imgs.append((img_path, self.classes.index(cls)))  # index(cls)分别为0和1



    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert('RGB')
        if self.base_transform:
            image = self.base_transform(image)
        return image, label
