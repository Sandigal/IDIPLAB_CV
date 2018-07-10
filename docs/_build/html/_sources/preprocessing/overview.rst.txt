
数据预处理
==========

总览
----------

.. figure:: ../images/数据预处理/流程图.png
    :width: 600
    :alt: 流程图
    :align: center

    通用图像预处理流程

从各处收集到的图像往往需要进行处理才能送入模型进行训练，而`数据预处理`模块将解决相关问题。该模块将完成从读取图像到送入模型的主要功能。

上图给出了一种常见的数据预处理流程，包括数据增强和数据集分割两大块。

- **数据增强**：利用各种变换使扩充图像的多样性。

  模型越复杂，所需的数据就越多。数据约少就越容易产生过拟合。在某些数据难以收集的问题上，可以采用 ``数据增强`` 的方法增广数据。``数据增强`` 完全免费，其目的使增强图像的多样性，从而提高模型的泛化能力。但 ``数据增强`` 并不能带来新的图像内容的，所以提升效果有限。

  该部分提供了目前常用的数据增强算法，包括翻转、噪声、裁剪、仿射变换等。也提供了数据增强库 imgaug_ 的接口，并且支持用户自定义数据增强算法。

.. _imgaug: https://github.com/aleju/imgaug

- **数据集分割**：让数据各司其职。

  收集到的数据并非全部都用于训练。一般简单的处理是将数据集分割为 **训练集**、**测试集**。其中，**训练集** 用于建立模型，**测试集** 用于评估选定模型的泛化情况。而另一个更严谨的方法是将数据集分割为 **训练集**、**验证集**、**测试集**。其中，验证集用于交叉检验和参数。

  用户可以随意调整数据集分割的比例。并且对于增强数据的归属问题，该部分也保证 **训练集** 中不会出现 **验证集** 和 **测试集** 对应的增强数据。

目录结构
----------

.. note:: 在不清楚 `python` 调用机制的情况下，我们强烈建议将工程文件按照下图进行存放。同时，以后所有的范例也默认为该存放方式。

`datasets`  只存放一个数据集，具体格式详见后文。``idiplab_cv`` 是本工程的所有库函数，可以在项目主页进行下载。`main` 是用户项目的主函数，所有功能的调用以及范例的使用将在主函数中进行进行。

::

    工程名/
    	datasets/
            ...
    	idiplab_cv/
            ...
    	main
    	...

`dataset` 文件夹中的图像必须按照以下格式存放。`dataset` 下的一级目录代表数据的来源，如origin是原始数据、augment是增强数据。二级目录代表了数据的类别信息，如dogs、cats。二级目录下的图像文件代表对应类别的样本，如dog001是dogs类的样本、cat001是cats类的样本。代码将直接从样本对应的路径读取从属类别以及类别名称。为了保证内存中的样本可以和文件中的样本一一对应，最好确保每个样本拥有不同名称。
::

    dataset/
    	origin/
            dogs/
                dog001.jpg
                dog002.jpg
                ...
            cats/
                cat001.jpg
                cat002.jpg
                ...
    		...
    	augment/
            dogs/
                dog001_XXXXX.jpg
                ...
            ...
    	...


应用范例
----------

此处将按照总览中的 **组合1** 和 **组合2** 进行举例说明。

组合1
````````

简单问题下，将原始数据的分割为 **训练集** 和 **测试集**。

1. 导入库函数。 ::

    from idiplab_cv.dataset_io import Dataset

2. 读取 `dataset` 文件夹内所有数据，得到各类对应代号和拥有的样本数。 ::

    dataset = io.Dataset(path="dataset")
    class_to_index, sample_per_class = dataset.load_data()


3. 进行 **训练集** 和 **测试集** 的分割。测试集大概拥有原始数据的20%。 ::

    imgs_train, labels_train, imgs_test, labels_test = dataset.train_test_split(test_shape=0.2)

4. 将标签转换为 `one-hot` 矩阵，并且进行 `shuffle`。 ::

    labels_train = io.label_str2index(labels_train, class_to_index)
    labels_train = io.to_categorical(labels_train, len(class_to_index))
    imgs_train, labels_train = shuffle(imgs_train, labels_train)

    labels_test = io.label_str2index(labels_test, class_to_index)
    labels_test = io.to_categorical(labels_test, len(class_to_index))



组合2
````````

包含 `数据增强`，将原始数据的分割为 **训练集**、**验证集** 和 **测试集**。并且在 **训练集** 和 **验证集** 上进行严谨的4折交叉检验。

1. 导入库函数。 ::

    from idiplab_cv.augment import agmt
    from idiplab_cv.dataset_io import Dataset

2. 对 `dataset` 文件夹内图像进行 ``数据增强``。相关变换包括翻转、裁剪、仿射变换。数据增广30倍。 ::

    datagen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=10.,
        zoom_range=0.1,
        channel_shift_range=5.,
        horizontal_flip=True
        )

    agmtgen = agmt.AugmentGenerator(path="dataset")
    agmtgen.normol_augment(datagen_args=datagen_args, augment_amount=30)

3. 如组合1进行数据读取以及 **训练集** 和 **测试集** 的分割。读取时包括增强数据。 ::

    dataset = io.Dataset(path="dataset", augment=True)
    class_to_index, sample_per_class = dataset.load_data()
    _, _, imgs_test, labels_test = dataset.train_test_split(test_shape=0.2)

4. 对 **训练集** 数据进行4折交叉检验。将 **训练集** 分为4组，每次取1组作为 **验证集** (20%)进行训练，其余作为原始 **训练集** (60%)。再读取原始 **训练集** (80%)对应的增强数据作为当前 **训练集** (660%)。
5. 如组合1将标签转换为 `one-hot` 矩阵，并且进行 `shuffle`。 ::

    total_splits = 4
    for valid_split in range(total_splits):
        imgs_train, labels_train, imgs_valid, labels_valid = dataset.cross_split(total_splits, valid_split)

        labels_train = io.label_str2index(labels_train, class_to_index)
        labels_train = io.to_categorical(labels_train, len(class_to_index))
        imgs_train, labels_train = shuffle(imgs_train, labels_train)

        labels_valid = io.label_str2index(labels_valid, class_to_index)
        labels_valid = io.to_categorical(labels_valid, len(class_to_index))


