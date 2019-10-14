import torch
import os
import argparse
import dihn
import adsh

from loguru import logger
from data.data_loader import load_data


def run():
    args = load_config()
    logger.add('logs/{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_seen,
        args.batch_size,
        args.num_workers,
    )

    # Training ADSH, create old binary hash code
    adsh.train(
        query_dataloader,
        seen_dataloader,
        retrieval_dataloader,
        args.code_length,
        args.device,
        args.lr,
        args.max_iter,
        args.max_epoch,
        args.num_samples,
        args.batch_size,
        args.root,
        args.dataset,
        args.gamma,
        args.topk,
    )

    # Increment learning
    B = torch.load(os.path.join('checkpoints', 'old_B.t'))
    mAP = dihn.increment(
        query_dataloader,
        unseen_dataloader,
        retrieval_dataloader,
        B,
        args.code_length,
        args.device,
        args.lr,
        args.max_iter,
        args.max_epoch,
        args.num_samples,
        args.batch_size,
        args.root,
        args.dataset,
        args.gamma,
        args.mu,
        args.topk,
    )
    logger.info('[map:{:.4f}]'.format(mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DIHN_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default=12, type=int,
                        help='Binary hash code length.(default: 12)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-seen', default=7, type=int,
                        help='Number of unseen classes.(default: 7)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--mu', default=50, type=float,
                        help='Hyper-parameter.(default: 50)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()
