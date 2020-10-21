![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------
# CuDNN Low-level APIs for PyTorch
This version of Pytorch is forked from the main PyTorch repo. There are some modifications to provide Python APIs for those who want to have more access to cuDNN backend. 

## Why did we need to cuDNN low-level APIs in PyTorch?
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers \[[1](https://developer.nvidia.com/cudnn)\]. 

PyTorch uses cuDNN as one of the backends to accelerate deep neural network training and inference. cuDNN provides APIs and different algorithms for each layer. As an example, cuDNN provides about 7 algorithms for convolution layer. Each of these algorithms has a difference execution time and memory usage. Choosing such parameteres affect both performance and energy consumption of Machine Learning (ML) applications in production. 

Among all of the possible cuDNN APIs, very few of them have been exposed to the user in PyTorch. For example, for convolution layer in ML applications, PyTorch only provides one of cuDNN APIs that based on convolution layer parameters (such as input size, kernel size, stride, dialation, and etc) uses either a `hueristic` or an `exhaustive` search approach to choose the best convolution algorithm (execution time by default). In current version of PyTorch, there is no cuDNN API for ML researchers and engineers to choose a specific implementation of convolution algorithms implemented in cuDNN. 

In this version of PyTorch, we expose some cuDNN APIs in PyTorch to give ML researchers more freedom to carefully choose and fine tune cuDNN parameters.


# Supported layers


## 1- Convolution Layer (forward)
PyTorch provides an API to choose between `hueristic` and `exhaustive` search approaches to perform convolution. By default, `hueristic` approach is enabled. In order to enable the `exhaustive` approach, the user has to `enbale` it by adding the below line to their Python script:

```torch.backends.cudnn.benchmark = True```

We expose an API to choose a convolution algorithm among the supported ones. In order to select the algorithm, add the following line to your Python script: 


```torch.backends.cudnn.conv_fwd_algo = N```

`N` is the number of the selected algorithm. In the latest version of cuDNN (8.0.4), there are 7 algorithms for convolution but not all of them are supported on all GPU architectures. In order to see which ones are supported on your target machine, the current version of this repo prints the supported convolution versions when the user performs a forward propagation with their netowrk. Below is the output of an script with one a network with one convolution layer. We selected algorithm #1 for convolution:
```
Conv FWD algo set to: 1             // output after executing: torch.backends.cudnn.conv_fwd_algo = 1
                                    // (printed by cuDNN)

FwdAlgorithms profile results:      // result of exhaustive search algorithm (printed by cuDNN)
Algo,   time, 	    memory
0,	    0.133632,	0
2,	    0.16272,	7840000
5,	    0.188288,	7468032
1,	    0.230432,	7424
7,	    0.603808,	20604480
4,	    1.02371,	13928000
6,	    -1,	        0
3,	    -1,	        0
----------------- 
Supported FwdAlgorithms:            // supported algorithms (printed by cuDNN)
0,  2,  5,  1,  7,  4
-----------------
Requested fwd algorithm (1) is set. //(printed by cuDNN)
```

# Tests Performed

The current version is tested with CUDA v10.01 and cuDNN v8.0.4


# Known issues / TODOs

## Version 0.1

1- Only forward propagation can be selected for now

2- The supported algorithms are unkown before running the `hueristic`/`exhaustive` algorithms. The supported algorithms are printed after running convolution layer. So the user needs to run the code ones and see the printed supported algorthms and then choose one from them.