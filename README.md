# pytorch-slimming

This is a **[PyTorch](http://pytorch.org/)** _re_-implementation of algorithm presented in "[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV2017)." . The official source code is based on Torch. For more info, visit the author's [webpage](https://github.com/liuzhuang13/slimming)!.

|  CIFAR10-VGG16BN  | Baseline | Trained with Sparsity (1e-4) | Pruned (0.7 Pruned) | Fine-tuned (40epochs) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  93.62   |            93.77             |        10.00        |         93.56         |
|    Parameters     |  20.04M  |            20.04M            |        2.42M        |         2.42M         |

|             Pruned Ratio             |       0       |     0.1      |      0.2      |     0.3      |     0.4      |     0.5      |     0.6      |     0.7      |
| :----------------------------------: | :-----------: | :----------: | :-----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| Top1 Accuracy (%) without Fine-tuned |     93.77     |    93.72     |     93.76     |    93.75     |    93.75     |    93.40     |    37.83     |    10.00     |
|       Parameters(M) / macc(M)        | 20.04/ 398.44 | 15.9/ 349.22 | 12.28/ 307.78 | 9.12/ 272.94 | 6.74/ 247.86 | 4.62/ 231.86 | 3.14/ 222.17 | 2.42/ 210.84 |

| Pruned Ratio |               architecture               |
| :----------: | :--------------------------------------: |
|      0       | [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512] |
|     0.1      | [60, 64, 'M', 128, 128, 'M', 256, 255, 253, 245, 'M', 436, 417, 425, 462, 'M', 463, 465, 472, 424] |
|     0.2      | [58, 64, 'M', 128, 128, 'M', 256, 255, 250, 233, 'M', 360, 336, 329, 398, 'M', 420, 412, 435, 341] |
|     0.3      | [56, 64, 'M', 128, 128, 'M', 256, 254, 249, 227, 'M', 284, 239, 244, 351, 'M', 369, 364, 384, 255] |
|     0.4      | [52, 64, 'M', 128, 128, 'M', 256, 254, 247, 218, 'M', 218, 162, 166, 294, 'M', 317, 315, 318, 165] |
|     0.5      | [52, 64, 'M', 128, 128, 'M', 256, 254, 245, 214, 'M', 179, 117, 116, 229, 'M', 228, 220, 210, 111] |
|     0.6      | [51, 64, 'M', 128, 128, 'M', 256, 254, 245, 213, 'M', 165, 85, 92, 153, 'M', 83, 86, 87, 111] |
|     0.7      | [49, 64, 'M', 128, 128, 'M', 256, 254, 234, 198, 'M', 114, 41, 24, 11, 'M', 14, 13, 19, 104] |

## Baseline 

```shell
python main.py
```

## Trained with Sparsity

```shell
python main.py -sr --s 0.0001
```

## Pruned

```shell
python prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.7
```

## Fine-tuned

```shell
python main.py -refine pruned.pth.tar --epochs 40
```

## Reference

```
@InProceedings{Liu_2017_ICCV,
    author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}
```
