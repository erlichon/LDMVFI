import random
import torch
import torchvision
import torchvision.transforms.functional as TF


def rand_crop(*args, sz):
    # import pdb; pdb.set_trace()
    # if args[0].size == (1024, 1024):
    #     for i in range(len(args)):
    #         args[i] = TF.resize(args[i], (512, 512))
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(args[0], output_size=sz)
    out = []
    for im in args:
        out.append(TF.crop(im, i, j, h, w))
    return out


def rand_flip(*args, p=0.5):
    out = list(args)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.hflip(im)
    if random.random() < p:
        for i, im in enumerate(out):
            out[i] = TF.vflip(im)
    return out

def rand_rotation(*args, p=0.5, max_degrees_per_step=3):
    out = list(args)
    rot_deg = random.uniform(-max_degrees_per_step, max_degrees_per_step)
    if random.random() < p:
        if random.random() < 0.5:
            for i, im in enumerate(out):
                out[i] = TF.rotate(im, rot_deg)
        else:
            for i, im in enumerate(out):
                out[i] = TF.rotate(im, i*rot_deg)
    return out

# def rand_continuous_translation(*args, p, max_step_size):
#     out = list(args)
#     rot_deg = random.randint(-max_degrees_per_step, max_degrees_per_step)
#     if random.random() < p:
#         for i, im in enumerate(out):
#             out[i] = TF.rotate(im, i*rot_deg)
#     return out

def rand_reverse(*args, p):
    if random.random() < p:
        return args[::-1]
    else:
        return args