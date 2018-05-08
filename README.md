# IDIPLAB_CV ![License](https://img.shields.io/aur/license/yaourt.svg?style=plastic)

IDIPLAB机器视觉工具包。



## 概括

IDIPLAB_CV是一个由IDIPLAB开发的用于机器视觉的顶层工具包，涵盖了基础的图形图像处理算法和集成化的深度学习框架接口。IDIPLAB_CV致力于将实验室的成功项目代码凝练成可以反复使用的工具包。

> 前人栽树后人乘凉，带领后辈走上科研第一线。

IDIPLAB_CV仅兼容python3。



## 如何使用

- 首先需要下载工具包。可以直接使用`git`命令获取全部工程文件，也可以通过[releases](https://github.com/Sandigal/IDIPLAB_CV/releases)下载代码。

  ```python
  git clone https://github.com/Sandigal/IDIPLAB_CV.git
  ```


- 接下来将名为**idiplab_cv**的文件夹放置到工程根目录内，导入相关函数即可使用。

  ```python
  import idiplab_cv
  ```


- 这时您的目录结构应该如下图所示。

  ```python
  工程名/
  	datasets/
  		数据集
  		...
  	idiplab_cv/
          相关函数
          ...
  	工程主函数
      ...
  ```



- 详细使用方法请阅读[**说明文档**](https://github.com/Sandigal/IDIPLAB_CV/wiki)。





## 依赖环境
所有代码均基于`python3`编写，并且需要如下第三方库支持。对于`windows`系统，强烈建议通过`anaconda`进行环境配置。

- [python](https://www.python.org/)

* [tensorflow](https://www.tensorflow.org/): 详细信息请阅读[官方安装指南](https://www.tensorflow.org/install/)。

  ```python
  pip install --upgrade tensorflow
  ```


* [Keras](https://keras.io/): 详细信息请阅读[官方安装指南](https://keras.io/#installation)。

  ```python
  pip install keras -U --pre
  ```


* **可选依赖**:
  -  [CUDA](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu) 和 [cuDNN](http://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04) (通过GPU加速运算)
  - HDF5 和 [h5py](http://docs.h5py.org/en/latest/build.html) (将Keras保存到本地)
  - [graphviz](https://graphviz.gitlab.io/download/) 和 [pydot](https://github.com/erocarrera/pydot) ( 使用[visualization utilities](https://keras.io/visualization/)进行模型可视化)




## 未来计划
* 更多`augmentation`方法
* 支持多尺寸图像
