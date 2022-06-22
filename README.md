# Arnold's Cat Map & Taskflow
This is an implementation of the [Arnold's Cat Map algorithm](http://fibonacci.math.uri.edu/~kulenm/diffeqaturi/victor442/index.html) using C++2017 and popular Graph Based Library [Taskflow](https://taskflow.github.io/), in the image bellow you can see the generated graph, also you can check the minipaper I wrote to my Parallel Algorithms Course `Propuesta_Paralelos.pdf`.  

![Arnold_cat_map_taskflow](/imgs/Arnold_Cat_Map_taskflow.png)

# What I need to compile it?
This is an heterogeneus type program, so it uses [CUDA](https://developer.nvidia.com/cuda-zone) to work with the GPU, here you have the list of required technologies to compile this program.

- C++ compiler (preferrable work in a Linux platform).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), you can find good tutorials on how to install it on Linux such as this [Tutorial](https://youtu.be/4gcqGxBIUnc).
- Download the [Taskflow](https://github.com/taskflow/taskflow) library hosted on github (just a `git clone` command).
- [stb_image](https://github.com/nothings/stb) library to work with images, but this already comes with the repo, so don't worry.
# How to compile it?
Once you have all requirements ready you can type the following command on your terminal to compile the program.  
```
$ nvcc -std=c++17 -I <path to taskflow repo> --extended-lambda --gpu-architecture=compute_<your compute GPU's capability> ArnoldTransform.cu -o ArnoldTransform
```
instructions between `<>` symbols are different dependent of your system, to find the *\<path to taskflow\>* just go to the folder you downloaded taskflow and type `pwd` on your terminal, to find *\<your GPU's compute capability\>* you can go to this [page](https://developer.nvidia.com/cuda-gpus) and find it depending on your GPU model just type `--gpu-architecture=compute_50` for example. You can find more info into the taskflow documentation, please visit these [section](https://taskflow.github.io/taskflow/CompileTaskflowWithCUDA.html).

# How to execute it?
Once you have your binary just give the name of the executable `ArnoldTransform` and the path for the image to be applied the Arnold's Cat Map, for example:  
```
$ ./ArnoldTransform imgs/test1/Lenna.jpg
```
***Very important!***  
***The program only accepts .jpg images***

# Display Outputs
Once you execute successfully the program after all the iterations you can check the result on terminal, in my case for example:  

```
saul@saul-ubuntu:~/Desktop/UCSP/Paralelos/Laboratorios/CUDA$ ./ArnoldTransform imgs/test1/Lenna.jpg 
Loaded image characteristics:
width: 512px
height: 512px
original N channels: 3
loaded with N channels: 3
Arnold Transformation ended
Image path: imgs/test1/Lenna.jpg
With 384 iterations
-----------------------------------------
digraph Taskflow {
subgraph cluster_p0x7ffd77a23700 {
label="Taskflow: Arnold's Cat Map Algorithm";
p0x55a9e5b6b570[label="resizer" ];
p0x55a9e5b6b570 -> p0x55a9e5b6b658;
p0x55a9e5b6b658[label="Helper" ];
p0x55a9e5b6b658 -> p0x55a9e5b6b828;
p0x55a9e5b6b658 -> p0x55a9e5b6b740;
p0x55a9e5b6b740[label="alloc_Pout" ];
p0x55a9e5b6b740 -> p0x55a9e5b6b910;
p0x55a9e5b6b828[label="alloc_Pin" ];
p0x55a9e5b6b828 -> p0x55a9e5b6b910;
p0x55a9e5b6b910[label="ArnoldTransformCudaFlow"  style="filled" color="black" fillcolor="purple" fontcolor="white" shape="folder"];
p0x55a9e5b6b910 -> p0x55a9e5b6b9f8;
subgraph cluster_p0x55a9e5b6b910 {
label="cudaFlow: ArnoldTransformCudaFlow";
color="purple"
p0x7f760c0115e0[label="Pin_h2d"];
p0x7f760c0115e0 -> p0x7f7610003100;
p0x7f76100032a0[label="Pout_d2h"];
p0x7f76100032a0 -> p0x55a9e5b6b910;
p0x7f7610003100[label="ArnoldKernel" style="filled" color="white" fillcolor="black" fontcolor="white" shape="box3d"];
p0x7f7610003100 -> p0x7f76100032a0;
}
p0x55a9e5b6b9f8[label="convergence&cls" shape=diamond color=black fillcolor=aquamarine style=filled];
p0x55a9e5b6b9f8 -> p0x55a9e5b6b658 [style=dashed label="0"];
p0x55a9e5b6b9f8 -> p0x55a9e5b6bae0 [style=dashed label="1"];
p0x55a9e5b6bae0[label="Finalizer" ];
}
}
```

The json format you see bellow is the generated tasks graph provided by taskflow you can visualize it with [GraphViz](https://dreampuf.github.io/GraphvizOnline/) online tool, here you have some outputs of the transformation:  
__Iteration 0 and 384__  
![Lenna_0](/imgs/Lenna.jpg)  
__Iteration 2__  
![Lenna_2](/imgs/Lenna_arnold_iter_2.jpg)  
__Iteration 307__  
![Lenna_307](/imgs/Lenna_arnold_iter_307.jpg)  
__Iteration 381__  
![Lenna_381](/imgs/Lenna_arnold_iter_381.jpg)
