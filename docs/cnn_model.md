# CNN Model

## Training

### Methodology

The backbone training is performend on a fixed number of epochs (1000). Later, we observe the loss decay via tensorboard, and pick the saved epochs where we start seeing a loss' plateau (in our case, from 700 to last). Next, we train a linear head over each one of these epochs while monitoring the validation performance. Thus, we select the weights were the validation accuracy is the highest, and with these weights, we evaluate the performance of the linear head using the testing set.

For DYB-linearHead dataset, **batch size 64**:

1. Backbone training (features extractor), using **ResNet50**. Elapsed training time 3 days, 21 hours. Use of memory: 11831MiB / 12288MiB (96.28%).
- The batch size was set to 64 to maximize the use of the GPU RAM while training, with the goal of minimizing the training time (let's see if this is true, performing now the training with batch size 32, let's see how log it takes...)

```bash
python main_supcon.py --batch_size 64 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0418, 0.0353, 0.0409" --std "0.0956, 0.0911, 0.0769" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --size 224
```

2. Head training (classifier). It is necessary to iterate over every backbone weights (in our case we iterate from epoch 700 to last):

```bash
./main_linear_loop.sh
```

---

For DYB-linearHead dataset, **batch size 32**:

1. Backbone training (features extractor), using **ResNet50**. Elapsed training time 4 days, 0 hours, 12 min. Use of memory: 7091MiB / 12288MiB (57.71%).

- The learning rate of 0.016 was selected as a linear relation between the original batch size of 1024 and learning rate of 0.5; thus dividing 0.5/32 = 0.0156 rounded to 0.016. 

```bash
python3 main_supcon.py --batch_size 32 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0418, 0.0353, 0.0409" --std "0.0956, 0.0911, 0.0769" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --size 224
```

2. Head training (classifier). It is necessary to iterate over every backbone weights (in our case we iterate from epoch 700 to last):

```bash
./main_linear_loop.sh
```

---

For DYB-linearHead dataset, **batch size 32**:

1. Backbone training (features extractor), using **SeResNext50**. Elapsed training time 6 days, 0 hours, 44 min. Use of memory 10371MiB / 12288MiB (84.40%).

```bash
python3 main_supcon.py --batch_size 32 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0418, 0.0353, 0.0409" --std "0.0956, 0.0911, 0.0769" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --size 224
```

2. Head training (classifier). It is necessary to iterate over every backbone weights (in our case we iterate from epoch 700 to last):

```bash
./main_linear.loop.sh
```

--- 

For DYB-cosine dataset, **batch size 32**:

1. Backbone training (features extractor):
```bash
python main_supcon.py --batch_size 32 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0419, 0.0355, 0.0410" --std "0.0959, 0.0913, 0.0771" --dataset path --data_folder /home/dsosatr/tesis/DYB-original/train/ --size 224
```

### Results

Resnet50timm, backbone batch_size = 64 and 32, learning_rate = 0.016, employed dataset DYB-linearHead, head trained using the frozen backbone on each epoch and validated with the DYB-original/val folder using a bs = 256 and learning_rate = 0.25. 17 jul 20:03.

SeResNext50timm, backbone batch_size = 32, learning_rate = 0.016, employed dataset DYB-linearHead, head trained using the frozen backbone on each epoch and validated with the DYB-original/val folder using a bs = 256 and learning_rate = 0.25. 6 ago 1:55

| | ResNet50 | ResNet50 | SeResNext50 |
| :-----------: | :----------------------: | :-----: | :-----: |
| ckpt number | ResNet50 ACC bs=64 / head epoch | ACC bs=32 / head epoch | ACC bs=32 / head epoch|
| last.pth    | 95.10% / 61   | 95.13% / 41 | 94.89% / 75 |
| 1000.pth    | 95.02% / 61   | 95.19% / 72 | 94.85% / 52 |
| 950.pth     | 95.03% / 51   | 95.19% / 44 | 94.85% / 56 |
| 900.pth     | 94.89% / 53   | 95.23% / 52 | 94.93% / 55 |
| 850.pth     | 94.88% / 66   | 95.23% / 53 | 94.91% / 48 |
| 800.pth     | 95.17% / 49   | 94.96% / 45 | 95.05% / 73 |
| 750.pth     | 94.64% / 70   | 94.92% / 56 | 94.89% / 58 |
| 700.pth     | 94.64% / 21   | 95.16% / 63 | 94.92% / 30 |

Finally, we test the accuracy of the backbone using the following batch sizes at epoch (see column "bs / backbone epoch / head epoch"), with the DYB-original/test dataset.

| bs / backbone epoch / head epoch | Accuracy | Precision | Recall | F1 Score | Network |
| ------------- | :------: | :-------: | :----: | :------: | :---: |
| 64 / 800 / 49 | 94.86%   | 93.65%    | 90.26% | 91.43%   | ResNet50|
| 32 / 900 / 52 | 95.10%   | 91.95%    | 89.87% | 90.39%   | ResNet50|
| 32 / 800 / 73 | 95.30%   | 93.83%    | 89.70% | 91.20%   | SeResNext50|

The above table was obtained using the following script, where we changed the input arguments model, val_folder, among others.
```bash
python3 main_check_accuracies.py
```

## Notes regarding training

[It is quite common for images with low resolution (e.g. 32x32 for CIFAR-10). In order to keep more spatial information before the average pool operator, the max pooling layer is removed. Notice that the filter-, stride- and padding size in the first convolution layer are also different from the conventional implementation. Again, this is to better preserve spatial information.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/74)

[In general you should scale the learning rate linearly with the batch size, such that a smaller batch size corresponds with a smaller learning rate. This is only valid for sgd though, adam has different rules.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/26)

## Additional references
- [SCL repository](https://github.com/HobbitLong/SupContrast)
- [Building LeNet5 in Pytorch](https://youtu.be/0TtYx_XaXjA?si=rX2Pzy1NJQ5G0ZRU)
- [LeNet5 git repo](https://github.com/maciejbalawejder/Deep-Learning-Collection/tree/main/ConvNets/LeNet)
- [CNN using Pytorch](https://medium.com/nerd-for-tech/convolution-neural-network-in-pytorch-81023e7de5b9)
- [CNN as feature extractor](cnn_theory.md)