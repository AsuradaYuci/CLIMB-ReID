import logging
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from .utils import *
from .loss import ClusterMemoryAMP, CrossEntropyLabelSmooth, TripletLoss


def train_climb(cfg,
              model,
              train_loader,
              val_loader,
              cluster_loader,
              optimizer,
              scheduler,
              num_query,
              num_classes):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("CLIMB")
    logger.info('start training')
    
    # model.to(device)
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = nn.DataParallel(model).cuda()

    loss_meter = AverageMeter()
    loss_meter1 = AverageMeter()
    loss_meter2 = AverageMeter()
    loss_meter3 = AverageMeter()
    loss_meter4 = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    xent = CrossEntropyLabelSmooth(num_classes)
    tri_loss = TripletLoss()
    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # scaler = amp.GradScaler()
    best_performance = 0
    best_epoch = 1
    # training epochs
    for epoch in range(1, epochs+1):
        loss_meter.reset()
        loss_meter1.reset()
        loss_meter2.reset()
        loss_meter3.reset()
        loss_meter4.reset()
        acc_meter.reset()
        acc_meter1.reset()

        evaluator.reset()

        # create memory bank
        image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
            
        num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
        logger.info(f'Memory has {num_classes} classes.')
        
        train_loader.new_epoch()
        
        # CAP memory
        memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info('Create memory bank with shape = {}'.format(memory.features.shape))
        
        # train one iteration
        model.train()
        num_iters = len(train_loader)
        for n_iter in range(num_iters):
            img, target, target_cam, _ = train_loader.next()
            
            optimizer.zero_grad()
            
            img = img.to(device)
            target = target.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # with amp.autocast(enabled=True):
            feat, logits, feat_sp, logits_sp = model(img, cam_label=target_cam, view_label=target_view)
            loss1 = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT
            # if cfg.MODEL.ID_LOSS_WEIGHT > 0:
            loss_id = xent(logits, target) * cfg.MODEL.ID_LOSS_WEIGHT
            loss_id2 = xent(logits_sp, target)
            loss_tri = tri_loss(feat_sp, target)
            loss = loss1 + loss_id + loss_id2 + loss_tri

            loss.backward()
            optimizer.step()

            # scaler.step(optimizer)
            # scaler.update()
            acc = (logits.max(1)[1] == target).float().mean()
            acc2 = (logits_sp.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_meter1.update(loss1.item(), img.shape[0])
            loss_meter2.update(loss_id.item(), img.shape[0])
            loss_meter3.update(loss_id2.item(), img.shape[0])
            loss_meter4.update(loss_tri.item(), img.shape[0])
            acc_meter.update(acc, 1)
            acc_meter1.update(acc2, 1)

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] "
                            "Loss_total: {:.3f}, "
                            "Loss1: {:.3f}, "
                            "Loss2: {:.3f}, "
                            "Loss3: {:.3f}, "
                            "Loss4: {:.3f}, "
                            "acc1: {:.3f},"
                            "acc2: {:.3f},"
                            "Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,
                                    loss_meter1.avg,
                                    loss_meter2.avg,
                                    loss_meter3.avg,
                                    loss_meter4.avg,
                                    acc_meter.avg,
                                    acc_meter1.avg,
                                    scheduler.get_lr()[0]))
        
        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))

        if epoch % eval_period == 0 and epoch >= 55:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camid.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    feat, feat1, feat2 = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, feat1, feat2, vid, camid))
            cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            logger.info("mAP_1: {:.1%}".format(mAP01))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc01[r - 1]))
            logger.info("mAP_2: {:.1%}".format(mAP02))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc02[r - 1]))
            logger.info("mAP_3: {:.1%}".format(mAP03))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc03[r - 1]))
            torch.cuda.empty_cache()
            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
            save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

            torch.cuda.empty_cache()
    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("CLIMB")
    logger.info("Enter inferencing")
    model.to(device)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()


    model.eval()
    for n_iter, (img, pid, camid, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]