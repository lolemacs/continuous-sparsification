# Continuous Sparsification

Implementation of Continuous Sparsification (CS), a method based on l_0 regularization to find sparse neural networks, proposed in [[Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427)].

## Requirements
```
Python 2/3, PyTorch == 1.1.0
```
## Training a ResNet on CIFAR with Continuous Sparsification

The main.py script can be used to train a ResNet-18 on CIFAR-10 with Continuous Sparsification. By default it will perform 2 rounds of training, each round consisting of 85 epochs. With the default hyperparameter values for the mask initialization, mask penalty, and final temperature, the method will find a sub-network with 20-30% sparsity which achieves 91.5-92.0% test accuracy when trained after rewinding (the dense network achieves 90-91%). The training and rewinding protocols follow the ones in the Lottery Ticket Hypothesis papers by Frankle.

In general, the sparsity of the final sub-network can be controlled by changing the value used to initialize the soft mask parameters. This can be done with, for example:

```
python main.py --mask-initial-value 0.1
```

The default value is 0.0 and increasing it will result in less sparse sub-networks. High sparsity sub-networks can be found by setting it to -0.1.

## Extending the code

To train other network models with Continuous Sparsification, the first step is to choose which layers you want to sparsify and then implement PyTorch modules that perform soft masking on its original parameters. This repository contains code for 2D convolutions with soft masking: the SoftMaskedConv2d module in models/layers.py:

```
class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, mask_initial_value=0.):
        super(SoftMaskedConv2d, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.init_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.init_mask()
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, temp, ticket):
        scaling = 1. / sigmoid(self.mask_initial_value)
        if ticket: mask = (self.mask_weight > 0).float()
        else: mask = F.sigmoid(temp * self.mask_weight)
        return scaling * mask      
        
    def prune(self, temp):
        self.mask_weight.data = torch.clamp(temp * self.mask_weight.data, max=self.mask_initial_value)   

    def forward(self, x, temp=1, ticket=False):
        self.mask = self.compute_mask(temp, ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding)        
        return out
        
    def checkpoint(self):
        self.init_weight.data = self.weight.clone()       
        
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
```

Extending it to other layers is straightforward, since you only need to change the init, init_mask and the forward methods. In init_mask, you should create a mask parameter (of PyTorch Parameter type) for each parameter set that you want to sparsify -- each mask parameter must have the same dimensions as the corresponding parameter.

```
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(...))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)
```

In the forward method, you need to compute the masked parameter for each parameter to be sparsified (e.g. masked weights for a Linear layer), and then compute the output of the layer with the corresponding PyTorch functional call (e.g. F.Linear for Linear layers). For example:

```
    def forward(self, x, temp=1, ticket=False):
        self.mask = self.compute_mask(temp, ticket)
        masked_weight = self.weight * self.mask
        out = F.linear(x, masked_weight)        
        return out
```

Once all the required layers have been implemented, it remains to implement the network which CS will sparsify. In models/networks.py, you can find code for the ResNet-18 and use it as base to implement other networks. In general, your network can inherit from MaskedNet instead of nn.Module and most of the required functionalities will be immediately available. What remains is to use the layers you implemented (the ones with soft masked paramaters) in your network, and remember to pass temp and ticket as additional inputs: temp is the current temperature of CS (assumed to be the attribute model.temp in main.py), while ticket is a boolean variable that controls whether the parameters' masks should be soft (ticket=False) or hard (ticket=True). Having ticket=True means that the mask will be binary and the masked parameters will actually be sparse. Use ticket=False for training (i.e. sub-network search) and ticket=True once you are done and want to evaluate the sparse sub-network.

## Future plans

We plan to make the effort of applying CS to other layers/networks considerably smaller. This will be hopefully achieved by offering a function that receives a standard PyTorch Module object and returns another Module but with the mask parameters properly created and the forward passes overloaded to use masked parameters instead.

If there are specific functionalities that would help you in your research or in applying our method in general, feel free to suggest it and we will consider implementing it.

## Citation

If you use our method for research purposes, please cite our work:

```
@article{ssm2019cs,
       author = {Savarese, Pedro and Silva, Hugo and Maire, Michael},
        title = {Winning the Lottery with Continuous Sparsification},
      journal = {arXiv:1912.04427},
         year = "2019"
}
```

