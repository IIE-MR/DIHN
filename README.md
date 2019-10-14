# Deep Incremental Hashing Network for Efficient Image Retrieval

> NUS-WIDE implementation will be available later. Lack of gpu :( 

## REQUIREMENTS
1. pytorch>=1.0
2. loguru

## DATASETS
1. [CIFAR-10](https://pan.baidu.com/s/1baBOtVK2SKGRt1TKmx9ruA) Password: v1tj

~~2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3~~

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT] [--batch-size BATCH_SIZE]
              [--lr LR] [--code-length CODE_LENGTH] [--max-iter MAX_ITER]
              [--max-epoch MAX_EPOCH] [--num-seen NUM_SEEN]
              [--num-samples NUM_SAMPLES] [--num-workers NUM_WORKERS]
              [--topk TOPK] [--gpu GPU] [--gamma GAMMA] [--mu MU]

DIHN_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --batch-size BATCH_SIZE
                        Batch size.(default: 64)
  --lr LR               Learning rate.(default: 1e-4)
  --code-length CODE_LENGTH
                        Binary hash code length.(default: 12)
  --max-iter MAX_ITER   Number of iterations.(default: 50)
  --max-epoch MAX_EPOCH
                        Number of epochs.(default: 3)
  --num-seen NUM_SEEN   Number of unseen classes.(default: 7)
  --num-samples NUM_SAMPLES
                        Number of sampling data points.(default: 2000)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 0)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --gamma GAMMA         Hyper-parameter.(default: 200)
  --mu MU               Hyper-parameter.(default: 50)

  ```

## EXPERIMENTS

cifar-10: 7 original classes, 3 incremental classes.

nus-wide: 18 original classes, 3 incremental classes.

 | | 12 bits | 24 bits | 32 bits | 48 bits 
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
ADSH cifar-10 MAP@ALL | 0.6433 | 0.6434 | 0.6451 | 0.6424
+DIHN cifar-10 MAP@ALL | 0.9091 | 0.9117 | 0.9177 | 0.9217

~~ADSH nus-wide-tc21 MAP@5000 |~~

~~+DIHN nus-wide-tc21 MAP@5000 |~~

