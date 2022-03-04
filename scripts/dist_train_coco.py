import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import coco
from utils.losses import DenseEnergyLoss, get_aff_loss, get_energy_loss
from wetr.PAR import PAR
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)
from utils.optimizer import PolyWarmupAdamW
from wetr.model_attn_aff import WeTr

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/coco_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512,512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, edge, attn_pred = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)
            
            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=args.radius)
            valid_cam_resized = F.interpolate(resized_cam, size=(H,W), mode='bilinear', align_corners=False)
            aff_cam = propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask, cls_labels=cls_label, bkg_score=0.35)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)
            #infer_path_index = irnutils.PathIndex(radius=5, default_size=(edge.shape[2], edge.shape[3]))
            #irn_cams = irnutils.batch_propagate_edge(cams=resized_cam, edge=edge.detach(), cls_label=cls_label, path_index=infer_path_index)
            #irn_label = cam_to_label_irn(irn_cams.detach(), cls_label=cls_label, ignore_mid=False, cfg=cfg)
            ###

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aff_gts += list(aff_label.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:,0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=81)
    cam_score = evaluate.scores(gts, cams, num_classes=81)
    aff_score = evaluate.scores(gts, aff_gts, num_classes=81)
    model.train()
    return cls_score, seg_score, cam_score, aff_score

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def train(cfg):

    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = coco.CocoClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,)
    logging.info('\nNetwork config: \n%s'%(wetr))
    param_groups = wetr.get_param_groups()
    par = PAR(num_iter=15, dilations=[1,2,4,8,12,24])
    wetr.to(device)
    par.to(device)
    
    mask_size = int(cfg.dataset.crop_size // 16)
    infer_size = int((cfg.dataset.crop_size * max(cfg.cam.scales)) // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    attn_mask_infer = get_mask_by_radius(h=infer_size, w=infer_size, radius=args.radius)
    if args.local_rank==0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        dummy_input = torch.rand(1, 3, 384, 384).cuda(0)
        #writer.add_graph(wetr, dummy_input)
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0, ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))

    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)
        
        cls, segs, attns, attn_pred = wetr(inputs, seg_detach=args.seg_detach)

        cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

        ######################
        valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)

        aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.low_thre)
        aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.high_thre)
        aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        
        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
        refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
        refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
        refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

        aff_cam = aff_cam_l[:,1:]
        refined_aff_cam = refined_aff_cam_l[:,1:,]
        refined_aff_label = refined_aff_label_h.clone()
        refined_aff_label[refined_aff_label_h == 0] = cfg.dataset.ignore_index
        refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
        refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=cfg.dataset.ignore_index)
        ######################

        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)

        if n_iter <= 15000:
            refined_aff_label = refined_pseudo_label

        aff_label = cams_to_affinity_label(refined_aff_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        

        seg_loss = get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        #reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_aff_label, img_box=img_box, loss_layer=loss_layer)
        #seg_loss = F.cross_entropy(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)
        
        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss + 0.0 * seg_loss + 0.0 * aff_loss# + 0.0 * reg_loss
        else: 
            loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.1 * aff_loss# + 0.01 * reg_loss


        avg_meter.add({'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item(), 'aff_loss': aff_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1,).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            aff_gts = refined_aff_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)
            _, grid_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=aff_cam)
            _, grid_ref_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=refined_aff_cam)

            _attns_detach = [a.detach() for a in attns]
            _attns_detach.append(attn_pred.detach())
            #_, grid_ref_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=refined_valid_cam)
            #grid_attns_0 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0, n_row=cfg.train.samples_per_gpu)
            #grid_attns_1 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.3, n_row=cfg.train.samples_per_gpu)
            #grid_attns_2 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.6, n_row=cfg.train.samples_per_gpu)
            #grid_attns_3 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.9, n_row=cfg.train.samples_per_gpu)
            grid_attns = imutils.tensorboard_attn2(attns=_attns_detach, n_row=cfg.train.samples_per_gpu)

            grid_labels = imutils.tensorboard_label(labels=gts)
            grid_preds = imutils.tensorboard_label(labels=preds)
            grid_refined_gt = imutils.tensorboard_label(labels=refined_gts)
            grid_aff_gt = imutils.tensorboard_label(labels=aff_gts)
            #grid_irn_gt = imutils.tensorboard_label(labels=irn_gts)

            if args.local_rank==0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, aff_loss: %.4f, pseudo_seg_loss %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('aff_loss'), avg_meter.pop('seg_loss'), seg_mAcc))

                writer.add_image("train/images", grid_imgs, global_step=n_iter)
                writer.add_image("train/preds", grid_preds, global_step=n_iter)
                writer.add_image("train/pseudo_gts", grid_labels, global_step=n_iter)
                writer.add_image("train/pseudo_ref_gts", grid_refined_gt, global_step=n_iter)
                writer.add_image("train/aff_gts", grid_aff_gt, global_step=n_iter)
                #writer.add_image("train/pseudo_irn_gts", grid_irn_gt, global_step=n_iter)
                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
                writer.add_image("cam/aff_cams", grid_aff_cam, global_step=n_iter)
                writer.add_image("cam/refined_aff_cams", grid_ref_aff_cam, global_step=n_iter)

                writer.add_image("attns/top_stages_case0", grid_attns[0], global_step=n_iter)
                writer.add_image("attns/top_stages_case1", grid_attns[1], global_step=n_iter)
                writer.add_image("attns/top_stages_case2", grid_attns[2], global_step=n_iter)
                writer.add_image("attns/top_stages_case3", grid_attns[3], global_step=n_iter)

                writer.add_image("attns/last_stage_case0", grid_attns[4], global_step=n_iter)
                writer.add_image("attns/last_stage_case1", grid_attns[5], global_step=n_iter)
                writer.add_image("attns/last_stage_case2", grid_attns[6], global_step=n_iter)
                writer.add_image("attns/last_stage_case3", grid_attns[7], global_step=n_iter)

                writer.add_scalars('train/loss', {"seg_loss": seg_loss.item(), "cls_loss": cls_loss.item()}, global_step=n_iter)
                writer.add_scalar('count/pos_count', pos_count.item(), global_step=n_iter)
                writer.add_scalar('count/neg_count', neg_count.item(), global_step=n_iter)
                
        
        if (n_iter+1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth"%(n_iter+1))
            if args.local_rank==0:
                logging.info('Validating...')
                torch.save(wetr.state_dict(), ckpt_name)
            cls_score, seg_score, cam_score, aff_score = validate(model=wetr, data_loader=val_loader, cfg=cfg)
            if args.local_rank==0:
                logging.info("val cls score: %.6f"%(cls_score))
                logging.info("cams score:")
                logging.info(cam_score)
                logging.info("aff cams score:")
                logging.info(aff_score)
                logging.info("segs score:")
                logging.info(seg_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)
    
    ## fix random seed
    setup_seed(1)
    train(cfg=cfg)
