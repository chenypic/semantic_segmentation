# FCN

[论文](https://arxiv.org/pdf/1605.06211.pdf)

作者利用CNN自身, 获得 end-to-end, pixels-to-pixels 的分割效果. 作者的重要见解是建立“全卷积”网络，可以接受任意大小的输入，并产生相应大小的输出。作者将现有的分类网络（AlexNet, VGG, GooGleNet等）改为FCN, 并通过 fine-tuning 来迁移它们学习的参数, 以实现分割任务. 作者然后定义了一个 skip architecture 将深层粗糙的语义信息和浅层精细的外观信息综合起来，以产生精确和细致的分割结果。作者在PASCAL VOC, NYUDv2 和 SIFT-Flow 数据集中取得了最好的结果, 并且推论时间小于1/5秒.

