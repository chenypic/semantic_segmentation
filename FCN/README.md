# FCN

[论文](https://arxiv.org/pdf/1605.06211.pdf)

作者利用CNN自身, 获得 end-to-end, pixels-to-pixels 的分割效果. 作者的重要见解是建立“全卷积”网络，可以接受任意大小的输入，并产生相应大小的输出。作者将现有的分类网络（AlexNet, VGG, GooGleNet等）改为FCN, 并通过 fine-tuning 来迁移它们学习的参数, 以实现分割任务. 作者然后定义了一个 skip architecture 将深层粗糙的语义信息和浅层精细的外观信息综合起来，以产生精确和细致的分割结果。作者在PASCAL VOC, NYUDv2 和 SIFT-Flow 数据集中取得了最好的结果, 并且推论时间小于1/5秒.



CNN的强大之处在于它的多层结构能自动学习特征，并且可以学习到多个层次的特征：较浅的卷积层感知域较小，学习到一些局部区域的特征；较深的卷积层具有较大的感知域，能够学习到更加抽象一些的特征。这些抽象特征对物体的大小、位置和方向等敏感性更低，从而有助于识别性能的提高。

这些抽象的特征对分类很有帮助，可以很好地判断出一幅图像中包含什么类别的物体，但是因为丢失了一些物体的细节，不能很好地给出物体的具体轮廓、指出每个像素具体属于哪个物体，因此做到精确的分割就很有难度。

传统的基于CNN的分割方法的做法通常是：为了对一个像素分类，使用该像素周围的一个图像块作为CNN的输入用于训练和预测。这种方法有几个缺点：一是存储开销很大。例如对每个像素使用的图像块的大小为15x15，则所需的存储空间为原来图像的225倍。二是计算效率低下。相邻的像素块基本上是重复的，针对每个像素块逐个计算卷积，这种计算也有很大程度上的重复。三是像素块大小的限制了感知区域的大小。通常像素块的大小比整幅图像的大小小很多，只能提取一些局部的特征，从而导致分类的性能受到限制。

针对这个问题, UC Berkeley的Jonathan Long等人提出了Fully Convolutional Networks (FCN)用于图像的分割。该网络试图从抽象的特征中恢复出每个像素所属的类别。即从图像级别的分类进一步延伸到像素级别的分类。

通常CNN网络在卷积层之后会接上若干个全连接层, 将卷积层产生的特征图(feature map)映射成一个固定长度的特征向量。以AlexNet为代表的经典CNN结构适合于图像级的分类和回归任务，因为它们最后都期望得到整个输入图像的一个数值描述（概率），比如AlexNet的ImageNet模型输出一个1000维的向量表示输入图像属于每一类的概率(softmax归一化)。

FCN将传统CNN中的全连接层转化成一个个的卷积层。如下图所示，在传统的CNN结构中，前5层是卷积层，第6层和第7层分别是一个长度为4096的一维向量，第8层是长度为1000的一维向量，分别对应1000个类别的概率。FCN将这3层表示为卷积层，卷积核的大小(通道数，宽，高)分别为（4096,1,1）、（4096,1,1）、（1000,1,1）。所有的层都是卷积层，故称为全卷积网络。如下图所示。

![](https://raw.githubusercontent.com/chenypic/semantic_segmentation/master/image/FCN_1.png)

FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

最后逐个像素计算softmax分类的损失, 相当于每一个像素对应一个训练样本。下图是FCN的结构示意图：

![](https://raw.githubusercontent.com/chenypic/semantic_segmentation/master/image/FCN_2.png)



可以发现，经过多次卷积（还有pooling）以后，得到的图像越来越小,分辨率越来越低（粗略的图像），那么FCN是如何得到图像中每一个像素的类别的呢？为了从这个分辨率低的粗略图像恢复到原图的分辨率，FCN使用了上采样。例如经过5次卷积(和pooling)以后，图像的分辨率依次缩小了2，4，8，16，32倍。对于最后一层的输出图像，需要进行32倍的上采样，以得到原图一样的大小。

这个上采样是通过反卷积（deconvolution）实现的。对第5层的输出（32倍放大）反卷积到原图大小，得到的结果还是不够精确，一些细节无法恢复。于是Jonathan将第4层的输出和第3层的输出也依次反卷积，分别需要16倍和8倍上采样，结果就精细一些了。下图是这个卷积和反卷积上采样的过程： 













参考资料

1，http://blog.csdn.net/taigw/article/details/51401448

2，http://www.cnblogs.com/gujianhan/p/6030639.html

3，Zheng, Shuai, et al. “Conditional random fields as recurrent neural 
networks.” Proceedings of the IEEE International Conference on Computer 
Vision. 2015. 

4，Kamnitsas, Konstantinos, et al. “Efficient Multi-Scale 3D CNN with 
Fully Connected CRF for Accurate Brain Lesion Segmentation.” arXiv 
preprint arXiv:1603.05959 (2016).