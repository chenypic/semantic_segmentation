# semantic_segmentation



[A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review)

1. #### FCN [learning]

   - Fully Convolutional Networks for Semantic Segmentation
   - Submitted on 14 Nov 2014
   - [Arxiv Link](https://arxiv.org/abs/1411.4038)

   *Key Contributions*:

   - Popularize the use of end to end convolutional networks for semantic segmentation
   - Re-purpose imagenet pretrained networks for segmentation
   - Upsample using *deconvolutional* layers
   - Introduce skip connections to improve over the coarseness of upsampling

   *Explanation*:

   Key observation is that fully connected layers in classification networks can be viewed as convolutions with kernels that cover their entire input regions. This is equivalent to evaluating the original classification network on overlapping input patches but is much more efficient because computation is shared over the overlapping regions of patches. Although this observation is not unique to this paper (see [overfeat](https://arxiv.org/abs/1312.6229), [this post](https://plus.google.com/+PierreSermanet/posts/VngsFR3tug9)), it improved the state of the art on VOC2012 significantly.

   ![FCN architecture](http://blog.qure.ai/assets/images/segmentation-review/FCN%20-%20illustration.png)

   ​												Fully connected layers as a convolution. [Source](https://arxiv.org/abs/1411.4038).

   After convolutionalizing fully connected layers in a imagenet pretrained network like VGG, feature maps still need to be upsampled because of pooling operations in CNNs. Instead of using simple bilinear interpolation, *deconvolutional layers* can learn the interpolation. This layer is also known as upconvolution, full convolution, transposed convolution or fractionally-strided convolution.

   However, upsampling (even with deconvolutional layers) produces coarse segmentation maps because of loss of information during pooling. Therefore, shortcut/skip connections are introduced from higher resolution feature maps.

   *Benchmarks (VOC2012)*:

   | Score | Comment                               | Source                                   |
   | ----- | ------------------------------------- | ---------------------------------------- |
   | 62.2  | -                                     | [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s) |
   | 67.2  | More momentum. Not described in paper | [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=6103#KEY_FCN-8s-heavy) |

   *My Comments*:

   - This was an important contribution but state of the art has improved a lot by now though.