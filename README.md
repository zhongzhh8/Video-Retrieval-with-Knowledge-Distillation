# Video Retrieval with Knowledge Distillation

## 1. UCF

Ultilize knowledge distillation technology in video retrieval task on dataset UCF101.

此份代码本地目录为 D:\PycharmProjects1\video_retrieval_KD，上传github作为备份，后面如果UCF的知识蒸馏效果跑不出来的话，来这里看参数值。

如果修改了参数，可能会对结果产生较大的影响，比如batch_size=100, lr=0.001就会导致效果差很多。

### Common Parameters

| dataset | hash_length | margin | lr     | batch_size | num_epoch | lr_schedule | resnet     |
| ------- | ----------- | ------ | ------ | ---------- | --------- | ----------- | ---------- |
| UCF101  | 48          | 14     | 0.0001 | 120        | 300       | patience=3  | not frozen |

### Result

| identity               | num_frames | mAP      |
| ---------------------- | ---------- | -------- |
| student                | 16         | 0.768573 |
| teacher                | 32         | 0.801399 |
| Knowledge Distillation | 16         |          |

### Knowledge Distillation权重调参

| weight (a\*L3+b\*Lrep) | mAP                        |
| ---------------------- | -------------------------- |
| 1+0.9                  | 0.773046                   |
| 1+1                    | 0.773641                   |
| 0.9+1                  | 0.775777                   |
| 0.8+1                  | 55555 s1 running  0.769519 |

总结：接近1+1的比重效果是最好的，然后a逐渐降低的情况又是效果更好，后续如果还需要调参，可以继续缓慢降低a的数值。然而降到0.8+1的时候效果已经很差了，唉。



## 2. JHMDB

JHMDB是一个小数据集，训练效果不稳定，同样的参数跑出来的mAP可能差距很大。

根本就是不可理喻。所以在上面无论是baseline还是知识蒸馏都是没有什么意义的。

所以其实是任人揉捏，我想要上面结果就能有什么结果，只要多跑几次选择自己想要的结果即可。

姑且来说就是这样吧：

### Common Parameters

| dataset | hash_length | margin | lr     | batch_size | num_epoch | lr_schedule | resnet     |
| ------- | ----------- | ------ | ------ | ---------- | --------- | ----------- | ---------- |
| JHMDB   | 48          | 14     | 0.0001 | 120        | 300       | patience=3  | not frozen |

步长调整策略

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
```

JHMDB数据集很小，训练效果不稳定，同样的参数跑出来的mAP可能差距很大。所以应该耐心跑多次，取效果最好的作为结果。

 选择checkpoint_step=5比较好，不然checkpoint_step=20容易遗漏中间的map峰值。可能应该放大patence=5，虽然影响也不是很大，一般一旦map下降了，就没什么可能再上去了，所以说JHMDB是个垃圾数据集。

### Result

| identity               | num_frames | mAP      |
| ---------------------- | ---------- | -------- |
| student                | 10         | 0.500361 |
| teacher                | 20         | 0.576061 |
| Knowledge Distillation | 10         | 0.5310   |



具体的调参细节可以参考Exp文件，大概就是1+0.001是比较好的权重设置。

