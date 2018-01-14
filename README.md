# pytorch-slimming

This is a **[PyTorch](http://pytorch.org/)** reimplementation of algorithm presented in "[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)(ICCV2017)." . The source code is based on Torch. For more info, visit the author's [webpage](https://github.com/liuzhuang13/slimming)!.

|  CIFAR10-VGG16BN  | Baseline | Trained with Sparsity | Pruned(70% Pruned) | Fine-tuned |
| :---------------: | :------: | :-------------------: | :----------------: | :--------: |
| Top1 Accuracy (%) |  93.72   |         93.56         |         80         |            |
|    Parameters     |          |                       |                    |            |

## Baseline 

```python
python main.py
```

## Trained with Sparsity

```python
python main.py -sr --s 0.0001
```

## Pruned

```python
python prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.75
```

## Fine-tuned

```python
python main.py -refine pruned.pth.tar
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
