import json
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_from_paths
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    Either provide --dataroot with trainA/trainB subdirectories,
    or provide --split_file pointing to a JSON file with keys:
      trainA, trainB, testA, testB  (lists of absolute image paths).
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        if opt.split_file:
            with open(opt.split_file) as f:
                split = json.load(f)
            key_A = opt.phase + 'A'
            key_B = opt.phase + 'B'
            if key_A not in split or key_B not in split:
                raise KeyError(f"split_file must contain keys '{key_A}' and '{key_B}' for phase '{opt.phase}'")
            self.A_paths = sorted(make_dataset_from_paths(split[key_A], opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_from_paths(split[key_B], opt.max_dataset_size))
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            if opt.phase == "test" and not os.path.exists(self.dir_A) \
               and os.path.exists(os.path.join(opt.dataroot, "valA")):
                self.dir_A = os.path.join(opt.dataroot, "valA")
                self.dir_B = os.path.join(opt.dataroot, "valB")
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
