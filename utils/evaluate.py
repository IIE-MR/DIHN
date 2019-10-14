import torch


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP
