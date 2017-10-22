# FCN

[论文](https://arxiv.org/pdf/1605.06211.pdf)

作者利用CNN自身, 获得 end-to-end, pixels-to-pixels 的分割效果. 作者的重要见解是建立“全卷积”网络，可以接受任意大小的输入，并产生相应大小的输出。作者将现有的分类网络（AlexNet, VGG, GooGleNet等）改为FCN, 并通过 fine-tuning 来迁移它们学习的参数, 以实现分割任务. 作者然后定义了一个 skip architecture 将深层粗糙的语义信息和浅层精细的外观信息综合起来，以产生精确和细致的分割结果。作者在PASCAL VOC, NYUDv2 和 SIFT-Flow 数据集中取得了最好的结果, 并且推论时间小于1/5秒.

![](https://raw.githubusercontent.com/chenypic/semantic_segmentation/master/image/FCN_1.png)



通常CNN网络在卷积层之后会接上若干个全连接层, 将卷积层产生的特征图(feature map)映射成一个固定长度的特征向量。以AlexNet为代表的经典CNN结构适合于图像级的分类和回归任务，因为它们最后都期望得到整个输入图像的一个数值描述（概率），比如AlexNet的ImageNet模型输出一个1000维的向量表示输入图像属于每一类的概率(softmax归一化)。

FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

最后逐个像素计算softmax分类的损失, 相当于每一个像素对应一个训练样本。下图是FCN的结构示意图：

