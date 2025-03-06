import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.market1501 import Market1501
# from datasets.msmt17_v2 import MSMT17_V2
from datasets.msmt17 import MSMT17
from .preprocessing import RandomErasing
from .dataset import ImageDataset, IterLoader
from .sampler import RandomIdentitySampler, RandomMultipleGallerySampler

FACTORY = {
    'market1501': Market1501,
    'msmt17': MSMT17,
}


def make_CLIMB_dataloader(cfg, all_iters=False):
    """
    PCL dataloader. It returns 3 dataloaders: training loader, cluster loader and validation loader.
    - For training loader, PK sampling is applied to select K instances from P classes.
    - For cluster loader, a plain loader is returned with validation augmentation but on
      training samples.
    - For validation loader, a validation loader is returned on test samples.
    
    Args:
    - dataset: dataset object.
    - all_iters: if `all_iters=True`, number training iteration is decided by `num_samples//batchsize`
    """
    
    dataset = FACTORY[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    # train loader
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    train_set = ImageDataset(dataset.train, train_transforms)
    # sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    sampler = RandomMultipleGallerySampler(dataset.train, cfg.DATALOADER.NUM_INSTANCE)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers
    )
    train_loader = IterLoader(train_loader, cfg.SOLVER.ITERS if not all_iters else None)
    
    # val loader
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    num_queries = len(dataset.query)
    val_loader = DataLoader(
        val_set, batch_size=4096, shuffle=False, num_workers=num_workers
    )
    
    # cluster loader
    cluster_set = ImageDataset(dataset.train, val_transforms)
    cluster_loader = DataLoader(
        cluster_set, batch_size=4096, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num
