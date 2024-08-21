import numpy as np
import random
from os import listdir
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import ldm.data.vfitransforms as vt
from functools import partial

class Vimeo90k_triplet(Dataset):

    def __init__(self, db_dir, train=True,  crop_sz=(256,256), aug_flip=True, aug_reverse=True):
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.train = train
        if train:
            seq_list_txt = join(db_dir, 'tri_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'tri_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "im1.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index],  "im2.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index],  "im3.png"))

        if self.crop_sz is not None:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)

        if self.aug_flip:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5, p=0.5)
        
        if self.aug_reverse:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,256,3), 0-255
        
        
        if self.train:
            if np.random.rand() < 0.75:
                rot_option = np.random.randint(0,4)
                frame3 = np.rot90(frame3,rot_option)
                frame4 = np.rot90(frame4,rot_option)
                frame5 = np.rot90(frame5,rot_option)



        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}
    
    def __len__(self):
        return len(self.seq_path_list)



class Hydra_triplet(Dataset):
    IMG_DELIMITER = ";;;"
    def __init__(self, db_dir, train=True, crop_sz=(256,256), aug_flip=True, 
                 aug_reverse=True, aug_rot=True, aug_blur=True, samples_per_epoch=1000, test=False, val=False, small_dataset=False, resize_sz=(512, 512)):
        assert any([train, test, val]) and not all([train, test, val]) and train ^ test ^ val == 1,\
              "Only one of the parameters train, test, val can be set to True."
        self.crop_sz = crop_sz
        self.resize_sz = resize_sz
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.aug_rot = aug_rot
        self.aug_blur = aug_blur
        self.train = train
        suffix=""
        if small_dataset:
            suffix=".filtered"
        if train:
            seq_list_txt = join(db_dir, f'tri_trainlist.txt{suffix}')
        elif test:
            seq_list_txt = join(db_dir, f'tri_testlist.txt{suffix}')
        elif val:
            seq_list_txt = join(db_dir, f'tri_vallist.txt{suffix}')

        with open(seq_list_txt) as f:
            f_lines = [i.strip() for i in f.readlines() if i != '\n']
            number_of_tri = len(f_lines)
            if small_dataset:
                number_of_tri = 10000 if train else 2000
            contents = f_lines[:number_of_tri]
            seq_path = [line.strip() for line in contents if line != '\n']

        self.tri_seq_path_list = [[img_fpath for img_fpath in line.split(self.IMG_DELIMITER)] 
                                  for line in seq_path]
        
        self.samples_per_epoch = samples_per_epoch
    
    def load_image(self,img_path,transform=None):
        try:
            image = Image.open(img_path)
            if self.resize_sz:
                # image = image.resize(self.resize_sz, Image.LANCZOS)
                image = image.resize(self.resize_sz)
            # image = cv2.merge((image, image, image))
            if transform:
                image = transform(image)

        except Exception as e:
            print(img_path)
            print(e)
            print(image)
    
        return image
    def __getitem__(self, index):
        rawFrame3 = self.load_image(self.tri_seq_path_list[index][0])
        rawFrame4 = self.load_image(self.tri_seq_path_list[index][1])
        rawFrame5 = self.load_image(self.tri_seq_path_list[index][2])


        frame3, frame4, frame5 = self.augment(rawFrame3, rawFrame4, rawFrame5)

        # frames in Hydra dataset are greyscale with 16 bit per pixel
        frame3 = frame3/(32767.5) - 1.0
        frame4 = frame4/(32767.5) - 1.0
        frame5 = frame5/(32767.5) - 1.0
        # converting to 3 channels model is compatible with 1 channel
        frame3 = cv2.merge((frame3, frame3, frame3))
        frame4 = cv2.merge((frame4, frame4, frame4))
        frame5 = cv2.merge((frame5, frame5, frame5))

        # print(self.tri_seq_path_list[index])
        # print(frame3.shape, frame4.shape, frame5.shape)

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}
    
    def augment(self, rawFrame3, rawFrame4, rawFrame5):
        # if self.aug_reverse:
        #     rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)
        # if self.resize_sz:
        #     resize = transforms.Resize(self.resize_sz)
        #     rawFrame3, rawFrame4, rawFrame5 = resize(rawFrame3), resize(rawFrame4), resize(rawFrame5)
        if self.crop_sz is not None:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)
        if self.aug_flip:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5)
        
        to_array = partial(np.array, dtype=np.float32)
        if self.train:
            # continuous rotation and 90 degrees
            if self.aug_rot:
                rawFrame3, rawFrame4, rawFrame5 = vt.rand_rotation(rawFrame3, rawFrame4, rawFrame5)
            if self.aug_blur:
                # rawFrame3, rawFrame4, rawFrame5 = vt.rand_gaussian_blur(rawFrame3, rawFrame4, rawFrame5)
                pass
            
            frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(255, 255), 0-65535            
            if self.aug_rot:
                rot_option = np.random.randint(0,4)
                frame3 = np.rot90(frame3,rot_option)
                frame4 = np.rot90(frame4,rot_option)
                frame5 = np.rot90(frame5,rot_option)
        else:
            frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(255, 255), 0-65535

        return frame3, frame4, frame5

    # def load_image(self,img_path,transform=None):
    #     try:
    #         image = (np.array(Image.open(img_path)/8)).astype(np.uint8)
    #         image = cv2.merge((image, image, image))
    #         if transform:
    #             image = transform(image)

    #     except Exception as e:
    #         print(img_path)
    #         print(e)
    #     return image
    # def __getitem__(self, index):
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Resize(size=self.resize_sz, antialias=True), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #     )  # outptu tensor in [-1,1]
    #     rawFrame3 = self.load_image(self.tri_seq_path_list[index][0], transform=transform)
    #     rawFrame4 = self.load_image(self.tri_seq_path_list[index][1], transform=transform)
    #     rawFrame5 = self.load_image(self.tri_seq_path_list[index][2], transform=transform)
        
    #     frame3, frame4, frame5 = self.augment(rawFrame3, rawFrame4, rawFrame5)
    #     # print(self.tri_seq_path_list[index])
    #     print(frame3.shape, frame4.shape, frame5.shape)

    #     return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}
    
    # def augment(self, rawFrame3, rawFrame4, rawFrame5):
    #     # if self.aug_reverse:
    #     #     rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)
    #     if self.crop_sz is not None:
    #         rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)
    #     if self.aug_flip:
    #         rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5)
        
    #     if self.train:
    #         # continuous rotation and 90 degrees
    #         if self.aug_rot:
    #             rawFrame3, rawFrame4, rawFrame5 = vt.rand_rotation(rawFrame3, rawFrame4, rawFrame5)
    #         if self.aug_blur:
    #             # rawFrame3, rawFrame4, rawFrame5 = vt.rand_gaussian_blur(rawFrame3, rawFrame4, rawFrame5)
    #             pass
            
    #         # frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(255, 255), 0-65535            
    #         if self.aug_rot:
    #             rot_option = np.random.randint(0,4)
    #             frame3, frame4, frame5 = TF.rotate(rawFrame3, 90*rot_option, fill=0), TF.rotate(rawFrame4, 90*rot_option, fill=0), TF.rotate(rawFrame5, 90*rot_option, fill=0)
    #     else:
    #         pass
    #         frame3, frame4, frame5 = rawFrame3, rawFrame4, rawFrame5
    #         # frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(255, 255), 0-65535

    #     # print(frame3.shape)
    #     reshape_size = (self.crop_sz[0], self.crop_sz[1], 3)
    #     return frame3.reshape(reshape_size), frame4.reshape(reshape_size), frame5.reshape(reshape_size)
    #     # return frame3, frame4, frame5
    
    def __len__(self):
        return len(self.tri_seq_path_list)



class Vimeo90k_quintuplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), aug_flip=True, aug_reverse=True):
        seq_dir = join(db_dir, 'sequences')
        self.crop_sz = crop_sz
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse

        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'sep_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index],  "im1.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "im3.png"))
        rawFrame4 = Image.open(join(self.seq_path_list[index],  "im4.png"))
        rawFrame5 = Image.open(join(self.seq_path_list[index],  "im5.png"))
        rawFrame7 = Image.open(join(self.seq_path_list[index],  "im7.png"))

        if self.crop_sz is not None:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_crop(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, sz=self.crop_sz)

        if self.aug_flip:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_flip(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)
        
        if self.aug_reverse:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_reverse(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)

        frame1, frame3, frame4, frame5, frame7 = map(TF.to_tensor, (rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7))

        return frame1, frame3, frame4, frame5, frame7

    def __len__(self):
        return len(self.seq_path_list)

    
class BVIDVC_triplet(Dataset):
    def __init__(self, db_dir, res=None, crop_sz=(256,256), aug_flip=True, aug_reverse=True):
        db_dir = '/scratch/zl3958/VLPR/data'
        db_dir = join(db_dir, 'quintuplets')
        self.crop_sz = crop_sz
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir)]


    def __getitem__(self, index):

        cat = Image.open(join(self.seq_path_list[index], 'quintuplet.png'))

        rawFrame3 = cat.crop((256, 0, 256*2, 256)) 
        rawFrame5 = cat.crop((256*2, 0, 256*3, 256))
        rawFrame4 = cat.crop((256*4, 0, 256*5, 256))

        if self.crop_sz is not None:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_crop(rawFrame3, rawFrame4, rawFrame5, sz=self.crop_sz)

        if self.aug_flip:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_flip(rawFrame3, rawFrame4, rawFrame5, p=0.5)
        
        if self.aug_reverse:
            rawFrame3, rawFrame4, rawFrame5 = vt.rand_reverse(rawFrame3, rawFrame4, rawFrame5, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame3, frame4, frame5 = map(to_array, (rawFrame3, rawFrame4, rawFrame5)) #(256,256,3), 0-255

        if np.random.rand()<0.75:
            rot_option = np.random.randint(0,4)
            frame3 = np.rot90(frame3,rot_option)
            frame4 = np.rot90(frame4,rot_option)
            frame5 = np.rot90(frame5,rot_option)

        frame3 = frame3/127.5 - 1.0
        frame4 = frame4/127.5 - 1.0
        frame5 = frame5/127.5 - 1.0

        return {'image': frame4, 'prev_frame': frame3, 'next_frame': frame5}

    def __len__(self):
        return len(self.seq_path_list)


class BVIDVC_quintuplet(Dataset):
    def __init__(self, db_dir, res=None, crop_sz=(256,256), aug_flip=True, aug_reverse=True):

        db_dir = join(db_dir, 'quintuplets')
        self.crop_sz = crop_sz
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir)]

    def __getitem__(self, index):

        cat = Image.open(join(self.seq_path_list[index], 'quintuplet.png'))

        rawFrame1 = cat.crop((0, 0, 256, 256))
        rawFrame3 = cat.crop((256, 0, 256*2, 256))
        rawFrame5 = cat.crop((256*2, 0, 256*3, 256))
        rawFrame7 = cat.crop((256*3, 0, 256*4, 256))
        rawFrame4 = cat.crop((256*4, 0, 256*5, 256))

        if self.aug_flip:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_flip(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)
        
        if self.aug_reverse:
            rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7 = vt.rand_reverse(rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7, p=0.5)

        frame1, frame3, frame4, frame5, frame7 = map(TF.to_tensor, (rawFrame1, rawFrame3, rawFrame4, rawFrame5, rawFrame7))

        return frame1, frame3, frame4, frame5, frame7

    def __len__(self):
        return len(self.seq_path_list)


class Sampler(Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch


class BVI_Vimeo_triplet(Dataset):
    def __init__(self, db_dir, crop_sz=[256,256], p_datasets=None, iter=False, samples_per_epoch=1000):
        vimeo90k_train = Vimeo90k_triplet(join(db_dir, 'vimeo_septuplet'), train=True,  crop_sz=crop_sz)
        bvidvc_train = BVIDVC_triplet(join(db_dir, 'bvidvc'), crop_sz=crop_sz)

        self.datasets = [vimeo90k_train]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch
        
