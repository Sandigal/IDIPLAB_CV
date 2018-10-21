总览
====================


下载别人的工程，运行后效果不错，然后想魔改到自己的任务上。你可能会发现效果并没有官方的那么好，甚至就像一切都还是随机值一样。那么你可能忽略了工程中对特定数据的适配问题。为此我们总结了一些常见的问题。

训练集时准确率和测试时不一样？
====================

这个问题经常发生在更换原始工程的数据集后。总体来讲原因有以下几种：

1. 训练集和测试集没有做相同的预处理。

2. 训练次数太少，某些层的权值未更新。

3. windows和linux下，文件的读取顺序不同。

训练集和测试集没有做相同的预处理
--------------------

不得不说，在进行各种工程时，经常忘记保持对测试集进行相同的预处理。这里的处理包括标准化、归一化、ZCA白化等预处理。在你对训练集进行如上操作后，切记保存从训练集中提取的均值、标准差等参数，然后用相同的参数处理测试集。

例如如果你使用了Keras中的ImageDataGenerator函数来进行图像预处理，记得使用相同的实例来处理训练集和测试集。

::

	# 训练时
	datagen = ImageDataGenerator(featurewise_center=True) # 定义方法
	datagen.fit(x_train) # 对训练集采样
	datagen.flow(x_train, y_train) # 处理训练集
	mean = datagen.mean # 保存参数

	# 测试时
	datagen2 = ImageDataGenerator(featurewise_center=True) # 定义方法
	datagen2.mean = mean # 加载训练集的参数
	datagen2.flow(x_test, y_test) # 使用相同参数处理测试集

回归	

.. note:: ImageDataGenerator.flow_from_directory永远无法处理基于对数据集采样的预处理方法，就算按照上述例子加载参数也无法顺利执行。你需要使用preprocessing_function来手动进行预处理操作。相关问题例子可以参照 Keras Issues #3679_ 6121_ 7218_。

	::
		
		def featurewise_center(x):
		    x = x-mean
		    return x

.. _#3679: https://github.com/keras-team/keras/issues/3679
.. _6121: https://github.com/keras-team/keras/issues/6121
.. _7218: https://github.com/keras-team/keras/issues/7218


训练次数太少，某些层的权值未更新
--------------------