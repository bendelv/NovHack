# Convolution
First, we need to understand a standard 2D convolution operation. A standard 2D convolution applies 2D filters/kernels over the inputs at **fixed perceptive fild and spacial locations** to generate the output feature map. Certain parameters are involved in the convolution opération named as kernal, stride and padding.

## Kernel size:
Defines the fieldof view of the convolution.

## Stride:
Defines the step size of the kernels when traversing the image.

## PAdding:
Defines how the border of sample is handled. For instance, for a kernel of 3 with stride of 1, no padding would result in down-smpling.

### 2D convilution using a kernel size of 3, stride of 1 and no padding
<figure>
  <p align="center">
  <img src=../Images/conv.gif  with=30%/>
  </p>
</figure>

## Convolution with padding

From the simplest convolution maps, we can see that after convolution operation, the output will be smaller than the input, but sometimes we want the output size to be consistent with the input, and padding is introduced for this purpose, and this kind of padding in order to keep the input and output sizes the same, we will call it “the same paddling”. The actual performance of padding is to add 0 around the input, and the upper limit is the size of the convolution core – 1. As in the dotted line area in the figure below, generally speaking, the size of padding will not be given in the paper, and we need ourselves. Derivation, deduction formula can be seen below.
2D convilution using a kernel size of 3, stride of 1 and padding of 1


<figure>
  <p align="center">
  <img src=../Images/convpadding1.gif  with=30%/>
  </p>
</figure>

- Same padding: Padding added to make the output the same as the input size, such as 3The nucleus of 3, same padding = 1,5The nucleus of 5, same padding = 2.
- full padding: padding = kernel size – 1
- valid padding: padding = 0

## Convolution of 2.3 stride greater than 1

Stride is the step size, which means the distance between two convolution operations of the convolution core. The default is 1. The two examples mentioned above are both steps 1. The following two graphs show the case of stride 2, which is the case without padding and with padding, respectively. Usually when the stride is greater than 1, we call it isometric downsampling, because the output will definitely lose information, and the size is smaller than the input.

### Convolution with no padding and strike = 2
<figure>
  <p align="center">
  <img src=../Images/convNopaddingStrike2.gif  with=50%/>
  </p>
</figure>



### TO TRY  / KEEP IN MIND / TIPS ON DESIGNING AN ARCHITECTURE

[link1]:(https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)


1. Always start by using smaller filters is to collect as much local information as possible, and then gradually increase the filter width to reduce the generated feature space width to represent more global, high-level and representative information

2. Following the principle, the number of channels should be low in the beginning such that it detects low-level features which are combined to form many complex shapes(by increasing the number of channels) which help distinguish between classes.

3. General filter sizes used are 3x3, 5x5 and 7x7 for the convolutional layer for a moderate or small-sized images and for Max-Pooling parameters we use 2x2 or 3x3 filter sizes with a stride of 2. Larger filter sizes and strides may be used to shrink a large image to a moderate size and then go further with the convention stated.

4. Try using padding = same when you feel the border’s of the image might be important or just to help elongate your network architecture as padding keeps the dimensions same even after the convolution operation and therefore you can perform more convolutions without shrinking size.

5. Keep adding layers until you over-fit. As once we achieved a considerable accuracy in our validation set we can use regularization components like l1/l2 regularization, dropout, batch norm, data augmentation etc. to reduce over-fitting

[Link2]:(https://datascience.stackexchange.com/questions/20222/how-to-decide-neural-network-architecture)

1. Create a network with hidden layers similar size order to the input, and all the same size, on the grounds that there is no particular reason to vary the size (unless you are creating an autoencoder perhaps).

2. Start simple and build up complexity to see what improves a simple network.

3. Try varying depths of network if you expect the output to be explained well by the input data, but with a complex relationship (as opposed to just inherently noisy).

4. Try adding some dropout, it's the closest thing neural networks have to magic fairy dust that makes everything better (caveat: adding dropout may improve generalisation, but may also increase required layer sizes and training times).

https://github.com/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb




TO TRY ->
- increase the complexity of the network bit by bit
- use padding to have convolution focus on edges of the boards
- stop using pooling (at least at first layers) to avoid locality invariance
- dropout : increase complexity of network until overfit, then increase drop out rate, and repeat
