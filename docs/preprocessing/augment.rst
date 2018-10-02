
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

:ref:`augment` 将完成通用的非监督数据增强方法。

augment
--------------------

.. automodule:: augment
    :members:
    :undoc-members:
    :show-inheritance:

.. _数据增强方法:
    
数据增强方法
--------------------
    
:meth:`AugmentGenerator.normol_augment` 目前支持以下 ``数据增强`` 方法。对于指定参数的方法，如果为 `rotation_range=30`，在实际运算时，具体参数将在0-30之间随机选取。
    
    - **featurewise_center**: :obj:`bool`。将输入数据的均值设置为 0，逐特征进行。
    - **samplewise_center**: :obj:`bool`。将每个样本的均值设置为 0。
    - **featurewise_std_normalization**: 布尔值。将输入除以数据标准差，逐特征进行。
    - **samplewise_std_normalization**: 布尔值。将每个输入除以其标准差。
    - **zca_epsilon**: ZCA 白化的 epsilon 值，默认为 1e-6。
    - **zca_whitening**: :obj:`bool`。应用 ZCA 白化。
    - **rotation_range**: 整形数。随机旋转的度数范围。
    - **width_shift_range**: 浮点数、一维数组或整数。随机水平移动的范围。
        - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
        - 1-D 数组: 数组中的随机元素。
        - int: 来自间隔 `(-width_shift_range, +width_shift_range)` 之间的整数个像素。
        - `width_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `width_shift_range=[-1, 0, +1]` 相同；而 `width_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
    - **height_shift_range**: 浮点数、一维数组或整数。随机垂直移动的范围。参数使用同 **width_shift_range**。
    - **shear_range**: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    - **zoom_range**: 浮点数或[lower, upper]。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
    - **channel_shift_range**: 浮点数。随机通道转换的范围。
    - **fill_mode**: {"constant", "nearest", "reflect" or "wrap"} 之一。默认为'nearest'。输入边界以外的点根据给定的模式填充：
        - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
        - 'nearest': aaaaaaaa|abcd|dddddddd
        - 'reflect': abcddcba|abcd|dcbaabcd
        - 'wrap': abcdabcd|abcd|abcdabcd
    - **cval**: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
    - **horizontal_flip**: 布尔值。随机水平翻转。
    - **vertical_flip**: 布尔值。随机垂直翻转。
    - **rescale**: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
    - **preprocessing_function**: 自定义 ``数据增强`` 算法的接口。具体介绍可以参见参照 :ref:`自定义方法`。
    - **data_format**: 图像数据格式，{"channels_first", "channels_last"} 之一。"channels_last" 模式表示图像输入尺寸应该为 `(samples, height, width, channels)`，"channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
    
.. _自定义方法:
    
自定义方法
--------------------

除了在 :ref:`自定义方法` 提到的各种方法，该模块还提供了自定义 ``数据增强`` 方法的接口。将编写的函数作为参数传入 **preprocessing_function** 中，这个函数会在任何其他改变 *之前* 运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。例如 ::

    def noise(image):
        height, width = image.shape[:2]
        for i in range(int(0.0005*height*width)):
            x = np.random.randint(0, height)
            y = np.random.randint(0, width)
            image[x, y, :] = 255
        return image

    preprocessing_function = noise
    
注意到上述方法无法控制内部参数，适合简单的 ``数据增强`` 方法。对于多变而复杂的方法，推荐使用 :obj:`class` 的方式编写方法。例如 ::

    class noise(object):

        def __init__(self, amount):
            self.amount = amount

        def __call__(self, img):
            height, width = image.shape[:2]
            for i in range(int(self.amount*height*width)):
                x = np.random.randint(0, height)
                y = np.random.randint(0, width)
                image[x, y, :] = 255
            return img

    preprocessing_function = noise(0.0005)

``python`` 本身就包含了很多图像处理工具包，而 **preprocessing_function** 的设置可以将这些工具包用到 ``数据增强`` 中。为此 :ref:`preprocess` 已经提供了一些常用工具包的使用接口。

preprocess
--------------------

.. automodule:: preprocess
    :members:
    :undoc-members:
    :show-inheritance:





    