import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import glob
import os
from torchvision import transforms


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames


class fusion_dataset_gt(Dataset):
    def __init__(self, ir_path=None, vis_path=None, gt_path=None):
        super(fusion_dataset_gt, self).__init__()
        self.filepath_vis, self.filenames_vis = prepare_data_path(vis_path)
        self.filepath_ir, self.filenames_ir = prepare_data_path(ir_path)
        self.filepath_gt, self.filenames_gt = prepare_data_path(gt_path)
        self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]
        gt_path = self.filepath_gt[index]
        
        image_vis = Image.open(vis_path)
        image_ir = Image.open(ir_path).convert('L')  # 转换为灰度图
        image_gt = Image.open(gt_path)
        
        image_vis = self.to_tensor(image_vis)
        image_ir = self.to_tensor(image_ir)
        image_gt = self.to_tensor(image_gt)
        
        name = self.filenames_vis[index]

        return image_vis, image_ir, image_gt, name

    def __len__(self):
        return self.length
