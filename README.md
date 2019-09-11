# Video Retrieval with Knowledge Distillation

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

| weight (a\*L3+b\*Lrep) | mAP              |
| ---------------------- | ---------------- |
| 1+0.9                  | 0.773046         |
| 1+1                    | 0.773641         |
| 0.9+1                  | 0.775777         |
| 0.8+1                  | 55555 s1 running |

总结：接近1+1的比重效果是最好的，然后a逐渐降低的情况又是效果更好，后续如果还需要调参，可以继续缓慢降低a的数值。



**重跑baseline**

| identity | num_frames | mAP                            |
| -------- | ---------- | ------------------------------ |
| student  | 16         | 55556 s1 running    e90  0.751 |
| teacher  | 32         | 55556 s2 running    e50  0.745 |

