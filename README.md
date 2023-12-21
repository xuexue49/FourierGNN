# FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

This repo is the official Pytorch implementation of ["FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective"](https://arxiv.org/pdf/2311.06190.pdf).

### Running the Codes
`python main.py`

- Covid: --feature_size 55 --embed_size 256 --hidden_size 512 -- batch_size 4 --train_ratio 0.6 --val_ratio 0.2
- METR-LA: --data metr --feature_size 207 --embed_size 128 --hidden_size 256 --batch_size 32 --train_ratio 0.7 --val_ratio 0.2
- Traffic: --feature_size 963 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- ECG: --data ECG --feature_size 140  --embed_size 128 --hidden_size 256 --batch_size 4 --train_ratio 0.7 --val_ratio 0.2
- Solar: --feature_size 592 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- Wiki: --feature_size 2000 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- Electricity: --feature_size 370 --embed_size 128 --hidden_size 256 --batch_size 32 --train_ratio 0.7 --val_ratio 0.2

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{yi2023fouriergnn,
title={Fourier{GNN}: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective},
author={Kun Yi and Qi Zhang and Wei Fan and Hui He and Liang Hu and Pengyang Wang and Ning An and Longbing Cao and Zhendong Niu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

1. StemGNN: https://github.com/microsoft/StemGNN
2. MTGNN: https://github.com/nnzhan/MTGNN
3. GraphWaveNet: https://github.com/nnzhan/Graph-WaveNet
4. AGCRN: https://github.com/LeiBAI/AGCRN
