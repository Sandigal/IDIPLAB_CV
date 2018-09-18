# IDIPLAB_CV 

![AUR](https://img.shields.io/aur/license/yaourt.svg) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg) ![Documentation Status](https://readthedocs.org/projects/idiplab-cv/badge/?version=latest)

IDIPLAB机器视觉工具包。

## 概括

IDIPLAB_CV是一个由IDIPLAB开发的用于机器视觉的顶层工具包，涵盖了基础的图形图像处理算法和集成化的深度学习框架接口。我们致力于将实验室的成功项目代码凝练成可以反复使用的工具包。

**注意**IDIPLAB_CV并不是一个某种学习框架，您需要自行学习[tensorflow](https://github.com/tensorflow/tensorflow)、[CNTK](https://github.com/Microsoft/cntk)或者[Theano](https://github.com/Theano/Theano)等框架来进行模型的构建以及训练。就像在处理图像时都要进行预处理一样，从实际工程中拿到的数据不可能直接导入到框架中使用，而我们注重的就是为您提供这一阶段的辅助。此外，在模型训练之后，我们也为您提供了封装完善的分析模块，使用尽可能少的代码获得丰富的评价报告。

> 前人栽树后人乘凉，带领后辈跑步走上科研第一线。


## 配置环境

所有代码均基于`python3`编写，并且需要诸多第三方库支持。对于`windows`系统，强烈建议通过`anaconda`进行环境配置。

- [anaconda](https://www.anaconda.com/download/): 一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换。

- **第三方库**: 运行**idiplab_cv**所需要的基本库。

  ```powershell
  pip install -r requirements.txt
  ```

- **IDIPLAB_CV本身**:

    - 通过[releases](https://github.com/Sandigal/IDIPLAB_CV/releases)下载**idiplab_cv**。只包含工具包文件，并且所有模块均通过了严格的测试。

    - 使用`git`命令获取**idiplab_cv**。拥有全部工程文件，包括文档源码、最新功能以及单元测试。(目前推荐)

    ```powershell
    git clone https://github.com/Sandigal/IDIPLAB_CV.git
    ```

- **可选依赖**: 

    - **GPU环境**: **idiplab_cv**并不是深度学习框架，而CPU足以胜任模型的日常使用任务，所以并不依赖GPU加速。但如果您像自己训练模型，那么必须配置GPU环境。

        - [CUDA](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu)

        - [cuDNN](http://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04)

        - [tensorflow-gpu](https://www.tensorflow.org/install/)

    - [imgaug](https://github.com/aleju/imgaug): 强大的数据增强算法库。

        ```powershell
        pip install imgaug
        ```

        ​

## 如何使用

- 将名为`idiplab_cv`的文件夹放置到工程根目录内，导入相关函数即可使用。

  ```python
  import idiplab_cv
  ```

- **建议**您将工程的目录结构设置为下图形式，以便配合说明文档的应用范例。如果您对python的导入机制和相对路径有深刻理解，也可以自行排布目录结构。

  ```powershell
  CV/ # 工程名
  |-- datasets/ # 数据集
  | |-- ...
  |
  |-- idiplab_cv/ # 相关函数
  | |-- ...
  |
  |-- main # 工程主函数
  |-- ...
  ```



- 详细使用方法请阅读说明文档：[**GitHub Wiki（20%）**](https://github.com/Sandiagal/IDIPLAB_CV/wiki)、[**Read the Docs（30%）**](https://idiplab-cv.readthedocs.io/zh/latest/)。
- 遇见还没写入文档又急着利用的功能时，可以从工程的`test`文件夹中获得项目的测试范例，自行理解使用。

