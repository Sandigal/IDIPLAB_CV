
.. _数据增强:

数据增强
====================

模型越复杂，所需的数据就越多。数据约少就越容易产生过拟合。在某些数据难以收集的问题上，可以采用 ``数据增强`` 的方法增广数据。``数据增强`` 通过对有限的数据进行各种变换，从而产生更多的等价数据。比起人为收集更多数据，``数据增强`` 完全免费，并且可以增强图像的多样性，从而提高模型的泛化能力。但 ``数据增强`` 并不能带来新的图像内容的，所以提升效果有限。

.. figure:: ../images/preprocessing/heavy.jpg
    :alt: Heavy augmentations

    单张图像的数据增强范例

引用 imgaug_ 中的数据增强范例。我们对一只 **短尾矮袋鼠** 图像进行了仿射变换、颜色抖动、超像素解析、锐化平滑等变换后，产生了32张“新”图像。如第1行1列张图像，原先白色的袋鼠经过了数据增强，产生了粉红色的袋鼠。这将使网络获得 **颜色不变性**，即将不同颜色的袋鼠都识别为 **袋鼠**。类似的还有 **位置不变性**、**形状不变性** 等。此外如第1行3列张图像，袋鼠的头部被遮挡。这将 ``Dropout`` 的思想引入了 ``数据增强``，即通过对部分显著特征的限制，强迫网络学习更加一般化的特征。

.. _imgaug: https://github.com/aleju/imgaug

上述图像对于网络来说都是不同的输入，这样就通过各种变换增强了图像的多样性。但通过 ``数据增强`` 产生的数据并非能和真实的数据相提并论。比如第3行1列张图像，袋鼠图像被颠倒了，而在现实中几乎不会出现这样的袋鼠。所以网络从这样图像中学习到了什么，也难以利用到一般的袋鼠中。此外，毕竟 ``数据增强`` 是基于原始图像的再创作，并不能带来新的图像内容的。所以如果增广倍数太多，也会因为图相似度太高而使提升效果下降。。

``数据增强`` 可以大致分为2种：非监督数据增强、监督数据增强。

该模块将完成通用的数据增强方法。

API
--------------------

.. automodule:: augment
    :members:
    :undoc-members:
    :show-inheritance:

.. _数据增强方法:
    
数据增强方法
--------------------
    
:meth:`AugmentGenerator.normol_augment` 目前支持以下数据增强方法。对于指定参数的方法，如果为 `rotation_range=30`，在实际运算时，具体参数将在0-30之间随机选取。
    
    - **samplewise_center**: 布尔值。将每个样本的均值设置为 0。
    - **samplewise_std_normalization**: 布尔值。将每个输入除以其标准差。
    - **zca_epsilon**: ZCA 白化的 epsilon 值，默认为 1e-6。
    - **zca_whitening**: 布尔值。应用 ZCA 白化。
    - **rotation_range**: 整形数。随机旋转的度数范围。
    - **width_shift_range**: 浮点数（总宽度的比例）。随机水平移动的范围。
    - **height_shift_range**: 浮点数（总高度的比例）。随机垂直移动的范围。
    - **shear_range**: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    - **zoom_range**: 浮点数或[lower, upper]。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
    - **channel_shift_range**: 浮点数。随机通道转换的范围。
    - **fill_mode**: 
    {"constant", "nearest", "reflect" or "wrap"} 之一。
    
    输入边界以外的点根据给定的模式填充
        - "constant": `kkkkkkkk|abcd|kkkkkkkk` (`cval=k`)
        - "nearest": `aaaaaaaa|abcd|dddddddd`
        - "reflect": `abcddcba|abcd|dcbaabcd`
        - "wrap": `abcdabcd|abcd|abcdabcd`
    - **cval**: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
    - **horizontal_flip**: 布尔值。随机水平翻转。
    - **vertical_flip**: 布尔值。随机垂直翻转。
    - **rescale**: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
    - **preprocessing_function**: 自定义数据增强算法的接口。具体介绍可以参见参照 :ref:`自定义方法`。
    
.. _自定义方法:
    
自定义方法
--------------------


应用于每个输入的函数。这个函数会在任何其他改变之前运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。例如 ::

        def noise(image):
            height, width = image.shape[:2]
            for i in range(int(0.0005*height*width)):
                x = np.random.randint(0, height)
                y = np.random.randint(0, width)
                image[x, y, :] = 255
            return image
    


API2
--------------------

.. automodule:: preprocess
    :members:
    :undoc-members:
    :show-inheritance:





    