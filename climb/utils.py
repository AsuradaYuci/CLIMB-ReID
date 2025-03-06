import torch
import torch.nn.functional as F
from torch.cuda import amp
import tqdm
import random
import copy
import numpy as np
from collections import defaultdict
import torch
import shutil
import os.path as osp
import errno
import os


def pk_sampling(batchsize, k, pseudo_labels, samples):
    pseudo_labels = pseudo_labels.cpu()
    samples = samples.cpu()
    
    batch_idxs_dict = defaultdict(list)
    pids = torch.unique(pseudo_labels).cpu().tolist()
    
    for pid in pids:
        idxs = samples[pseudo_labels == pid].tolist()
        if len(idxs) < k:
            idxs = np.random.choice(idxs, size=k, replace=True)
        random.shuffle(idxs)
        batch_idxs = []
        for idx in idxs:
            batch_idxs.append(idx)
            if len(batch_idxs) == k:
                batch_idxs_dict[pid].append(batch_idxs)
                batch_idxs = []
    
    avai_pids = copy.deepcopy(pids)
    final_idxs = []
    
    while len(avai_pids) >= (batchsize // k):
        selected_pids = random.sample(avai_pids, batchsize//k)
        for pid in selected_pids:
            batch_idxs = batch_idxs_dict[pid].pop(0)
            final_idxs.extend(batch_idxs)
            if len(batch_idxs_dict[pid]) == 0:
                avai_pids.remove(pid)

    final_idxs = torch.split(torch.tensor(final_idxs), batchsize)
                
    return iter(final_idxs)


def extract_image_features(model, cluster_loader, use_amp=False):
    image_features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for _, (img, pid, camid, _) in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
            img = img.cuda()
            target = pid.cuda()
            camid = camid.cuda()
            # with amp.autocast(enabled=use_amp):
            image_feature = model(img, get_image = True,cam_label= camid)
            for i, img_feat in zip(target, image_feature):
                labels.append(i)
                image_features.append(img_feat.cpu())
    labels_list = torch.stack(labels, dim=0).cuda()
    image_features_list = torch.stack(image_features, dim=0).cuda() # NC
    return image_features_list, labels_list


def cam_label_split(cluster_labels, all_img_cams):
    """
    Split proxies using camera labels.
    """
    proxy_labels = -1 * torch.ones(cluster_labels.shape).type_as(cluster_labels)
    cnt = 0
    for i in range(0, int(cluster_labels.max() + 1)):
        inds = torch.where(cluster_labels == i)[0]
        local_cams = all_img_cams[inds]
        for cc in torch.unique(local_cams):
            pc_inds = torch.where(local_cams == cc)[0]
            proxy_labels[inds[pc_inds]] = cnt
            cnt += 1
    return proxy_labels

def compute_cluster_centroids(features, labels):
    """
    Compute L2-normed cluster centroid for each class.
    """
    num_classes = len(labels.unique()) - 1 if -1 in labels else len(labels.unique())
    centers = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
    for i in range(num_classes):
        idx = torch.where(labels == i)[0]
        temp = features[idx,:]
        if len(temp.shape) == 1:
            temp = temp.reshape(1, -1)
        centers[i,:] = temp.mean(0)
    return F.normalize(centers, dim=1)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))