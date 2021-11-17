import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
#from dataset import SceneTextDataset
from dataset import SceneTextDataset_Ex as SceneTextDataset
from model import EAST

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    dataset = SceneTextDataset(data_dir, split='split_train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    dataset_valid = SceneTextDataset(data_dir, split='split_valid', image_size=image_size, crop_size=input_size)
    dataset_valid = EASTDataset(dataset_valid)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    best_loss = 999
    for epoch in range(max_epoch):

        for stage in ['train', 'valid']:
            if stage == 'train':
                model.train()
                loader = train_loader
                color = 95
            elif stage == 'valid':
                model.eval()
                loader = valid_loader
                color = 96
            epoch_loss, epoch_start = 0, time.time()
            epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0., 0., 0.

            for img, gt_score_map, gt_geo_map, roi_mask in tqdm(loader):

                if stage == 'train':
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                elif stage == 'valid':
                    with torch.no_grad():
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                epoch_loss += loss.item()

                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

            epoch_loss /= len(loader)
            epoch_cls_loss /= len(loader)
            epoch_angle_loss /= len(loader)
            epoch_iou_loss /= len(loader)
            print(f'\n\033[{color}m {epoch}/{max_epoch} {stage}: '
                  f'Mean loss: {epoch_loss:.4f}, '
                  f'Cls loss: {epoch_cls_loss:.4f}, '
                  f'Angle loss: {epoch_angle_loss:.4f}, '
                  f'IoU loss: {epoch_iou_loss:.4f} | '
                  f'Elapsed time: {timedelta(seconds=time.time() - epoch_start)}' + '\033[0m')

            if stage == 'valid':
                # Best score 모델 저장
                if epoch_loss < best_loss:
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    ckpt_fpath = osp.join(model_dir, 'latest.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    print(f"Best performance at epoch: {epoch}, Save model in {ckpt_fpath}")
                    best_loss = epoch_loss

        scheduler.step()




def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
