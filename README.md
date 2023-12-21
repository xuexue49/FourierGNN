# FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

非官方修改,收集了两个数据集进行测试，结果与论文较符合，但似乎略优于论文中的数据。

METR-LA: MAPE 53.946374603%; MAE 0.048967090; RMSE 0.087826379.

ECG: MAPE 11.523763227%; MAE 0.050121520; RMSE 0.078611672.

### Running the Codes

修复了原readme中错误的运行参数

`python main.py`

- Covid: --feature_size 55 --embed_size 256 --hidden_size 512 -- batch_size 4 --train_ratio 0.6 --val_ratio 0.2
- METR-LA: --data metr --feature_size 207 --embed_size 128 --hidden_size 256 --batch_size 32 --train_ratio 0.7 --val_ratio 0.2
- Traffic: --feature_size 963 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- ECG: --data ECG --feature_size 140  --embed_size 128 --hidden_size 256 --batch_size 4 --train_ratio 0.7 --val_ratio 0.2
- Solar: --feature_size 592 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- Wiki: --feature_size 2000 --embed_size 128 --hidden_size 256 --batch_size 2 --train_ratio 0.7 --val_ratio 0.2
- Electricity: --feature_size 370 --embed_size 128 --hidden_size 256 --batch_size 32 --train_ratio 0.7 --val_ratio 0.2

## Citation

citation 请看原仓库

 
