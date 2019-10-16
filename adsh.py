import torch
import os
import time
import torch.optim as optim
import models.alexnet as alexnet
import utils.evaluate as evaluate

from torch.optim.lr_scheduler import ExponentialLR
from loguru import logger
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader
from dihn import generate_code


def train(
        query_dataloader,
        seen_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        dataset,
        gamma,
        topk,
):
    """
    Training model.

    Args
        query_dataloader, seen_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_samples(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        None
    """
    # Initialization
    model = alexnet.load_model(code_length).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    criterion = ADSH_Loss(code_length, gamma)
    lr_scheduler = ExponentialLR(optimizer, 0.9)

    num_seen = len(seen_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)
    B = torch.randn(num_seen, code_length).sign().to(device)
    seen_targets = seen_dataloader.dataset.get_onehot_targets().to(device)

    total_time = time.time()
    for it in range(max_iter):
        iter_time = time.time()
        lr_scheduler.step()

        # Sample training data for cnn learning
        train_dataloader, sample_index, _, _ = sample_dataloader(seen_dataloader, num_samples, num_seen, batch_size, root, dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
        S = (train_targets @ seen_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        # Training CNN model
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                F = model(data)
                U[index, :] = F.data
                cnn_loss = criterion(F, B, S[index, :], index)

                cnn_loss.backward()
                optimizer.step()

        # Update B
        expand_U = torch.zeros(B.shape).to(device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, gamma)

        # Total loss
        iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma)
        logger.debug('[iter:{}/{}][loss:{:.2f}][time:{:.2f}]'.format(it + 1, max_iter, iter_loss, time.time() - iter_time))

    logger.info('Training adsh finish, time:{:.2f}'.format(time.time()-total_time))

    # Save checkpoints
    torch.save(B.cpu(), os.path.join('checkpoints', 'old_B.t'))

    # Evaluate
    query_code = generate_code(model, query_dataloader, code_length, device)
    mAP = evaluate.mean_average_precision(
        query_code.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_dataloader.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    logger.info('[ADSH map:{:.4f}]'.format(mAP))


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()
