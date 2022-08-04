import argparse
import datetime
import os
import random
from collections import OrderedDict
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
#from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from wetr.model_attn_aff import WeTr
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument("--model_path", default="./wetr_iter_18000.pth", type=str, help="model_path")

def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds = [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, _ = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            #######
            # resize long side to 512
            _, _, h, w = inputs.shape
            ratio = args.resize_long / max(h,w)
            _h, _w = int(h*ratio), int(w*ratio)
            inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
            #######
            
            segs_list = []
            inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
            _, segs_cat, _, _ = model(inputs_cat, )
            segs = segs_cat[0].unsqueeze(0)

            _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
            segs_list.append(_segs)

            _, _, h, w = segs_cat.shape

            for s in test_scales:
                if s != 1.0:
                    _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                    _, segs_cat, _, _ = model(inputs_cat, )

                    _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                    _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                    segs_list.append(_segs)

            msc_segs = torch.max(torch.stack(segs_list, dim=0), dim=0)[0].unsqueeze(0)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_preds = torch.argmax(resized_segs, dim=1)

            resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
            _gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.cpu().numpy(), "msc_segs":msc_segs.cpu().numpy()})
            
    return _gts, _preds, _msc_preds


def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    score = evaluate.scores(gts, preds)

    print(score)
    
    return True

def main(cfg):
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,)
    
    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    wetr.load_state_dict(state_dict=new_state_dict, strict=True)
    wetr.eval()
    
    gts, preds, msc_preds = validate(model=wetr, dataset=val_dataset, test_scales=[1, 0.5, 0.75])
    torch.cuda.empty_cache()

    seg_score = evaluate.scores(gts, preds)
    msc_seg_score = evaluate.scores(gts, msc_preds)

    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    crf_proc(config=cfg)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    main(cfg=cfg)
