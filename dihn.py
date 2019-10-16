import torch
import torch.optim as optim
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate

from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR
from models.dihn_loss import DIHN_Loss
from data.data_loader import sample_dataloader


def increment(
        query_dataloader,
        unseen_dataloader,
        retrieval_dataloader,
        old_B,
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
        mu,
        topk,
):
    """
    Increment model.

    Args
        query_dataloader, unseen_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        old_B(torch.Tensor): Old binary hash code.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma, mu(float): Hyper-parameters.
        topk(int): Top k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    model = alexnet.load_model(code_length)
    model.to(device)
    model.train()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    criterion = DIHN_Loss(code_length, gamma, mu)
    lr_scheduler = ExponentialLR(optimizer, 0.91)

    num_unseen = len(unseen_dataloader.dataset)
    num_seen = len(old_B)
    U = torch.zeros(num_samples, code_length).to(device)
    old_B = old_B.to(device)
    new_B = torch.randn(num_unseen, code_length).sign().to(device)
    B = torch.cat((old_B, new_B), dim=0).to(device)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)

    total_time = time.time()
    for it in range(max_iter):
        iter_time = time.time()
        lr_scheduler.step()

        # Sample training data for cnn learning
        train_dataloader, sample_index, unseen_sample_in_unseen_index, unseen_sample_in_sample_index = sample_dataloader(retrieval_dataloader, num_samples, num_seen, batch_size, root, dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
        S = (train_targets @ retrieval_targets.t() > 0).float()
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
        expand_U = torch.zeros(num_unseen, code_length).to(device)
        expand_U[unseen_sample_in_unseen_index, :] = U[unseen_sample_in_sample_index, :]
        new_B = solve_dcc(new_B, U, expand_U, S[:, num_seen:], code_length, gamma)
        B = torch.cat((old_B, new_B), dim=0).to(device)

        # Total loss
        iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma, mu)
        logger.debug('[iter:{}/{}][loss:{:.2f}][time:{:.2f}]'.format(it + 1, max_iter, iter_loss, time.time() - iter_time))

    logger.info('[DIHN time:{:.2f}]'.format(time.time() - total_time))

    # Evaluate
    query_code = generate_code(model, query_dataloader, code_length, device)
    mAP = evaluate.mean_average_precision(
        query_code.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_targets,
        device,
        topk,
    )

    return mAP


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    P = code_length * S.t() @ U + gamma * expand_U

    for bit in range(code_length):
        p = P[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (p.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma, mu):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    correlation_loss = (U @ torch.ones(U.shape[1], 1, device=U.device)).sum()
    loss = (hash_loss + gamma * quantization_loss + mu * correlation_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
