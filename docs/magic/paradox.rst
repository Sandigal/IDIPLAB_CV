魔改须知一二
====================

下载别人的工程，运行后效果不错，然后想魔改到自己的任务上。你可能会发现效果并没有官方的那么好，甚至就像一切都还是随机值一样。那么你可能忽略了工程中对特定数据的适配问题。为此我们总结了一些常见的问题。

训练集时准确率和测试时不一样？
--------------------

这个问题经常发生在更换原始工程的数据集后。总体来讲原因有以下几种：

1. 训练集和测试集没有做相同的预处理。

2. 训练次数太少，某些层的权值未更新。

3. windows 和 linux 下，文件的读取顺序不同。

训练集和测试集没有做相同的预处理
````````````````````

不得不说，在进行各种工程时，经常忘记保持对测试集进行相同的预处理。这里的处理包括标准化、归一化、ZCA 白化等预处理。在你对训练集进行如上操作后，切记保存从训练集中提取的均值、标准差等参数，然后用相同的参数处理测试集。

例如如果你使用了 Keras 中的 ImageDataGenerator 函数来进行图像预处理，记得使用相同的实例来处理训练集和测试集。相关示例如下：

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

.. note:: ImageDataGenerator.flow_from_directory 永远无法处理基于对数据集采样的预处理方法，就算按照上述例子加载参数也无法顺利执行。你需要使用 preprocessing_function 来手动进行预处理操作。更多相关问题可以参照 Keras Issues 3679_、6121_、7218_。

	::
		
		def featurewise_center(x):
		    x = x-mean
		    return x

		ImageDataGenerator(preprocessing_function=featurewise_center)

.. _3679: https://github.com/keras-team/keras/issues/3679
.. _6121: https://github.com/keras-team/keras/issues/6121
.. _7218: https://github.com/keras-team/keras/issues/7218


训练次数太少，某些层的权值未更新
````````````````````

这里指的就是批标准化(Batch Normalization, BN)层。虽然BN层通过减少内部协变量偏移(Internal Covariate Shift)加速了网络训练，但是在实际使用时也多了一些需要注意的地方。

这个问题在新数据很小，并且训练次数也少的情况下影响很大。你需要知道 BN 层在训练与测试时的工作模式不同。训练时，均值和方差根据 mini-batch 计算；而测试时，均值和方差根据累计的均值和方差的滑动平均值计算（类似conv的更新方式）。

对于魔改的情况，首先 BN 层是需要训练的。由于你拿到的模型基本都是基于 Imagnet 训练的，BN 的均值和方差也是如此。而对于你的数据集，BN 层需要重新适应这些参数。要知道 BN 层的动量默认为0.99，这就需要相当的迭代次数才能完成参数的适配。

这有两种方法来从根源解决以上问题：

1. 更多迭代次数
	
	降低 batch size 可以在每 epoch 中获得更多权值更新机会。当然这样并行计算的优势就不明显了。或者单纯将 epochs 提高，就算收敛了也让网络继续训练。这样能够保证 BN 层能够收敛到正确的值，但就是耗费了额外的计算资源。

2. 修改 BN 层的动量
	
	保持 batch size、epochs 不变的情况下，修改动量可以让 BN 层更激进的收敛。就像提高学习率一样，过快的收敛可能并不能得到理想的结果，所以不建议在工业环境下使用该方法。对于 Keras，已经初始化的模型时无法调整动量的，你可以按照如下示例设置：

	::

		base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

		import json
		conf = json.loads(base_model.to_json())
		for l in conf['config']['layers']:
		    if l['class_name'] == 'BatchNormalization':
		        l['config']['momentum'] = 0.5

		m = Model.from_config(conf['config'])
		for l in base_model.layers:
		    m.get_layer(l.name).set_weights(l.get_weights())

		base_model = m


更多相关问题可以参照 Keras Issues 3679_、7177_、10014_。

.. _4762: https://github.com/keras-team/keras/issues/4762
.. _7177: https://github.com/keras-team/keras/issues/7177
.. _10014: https://github.com/keras-team/keras/issues/10014


windows 和 linux 下，文件的读取顺序不同。
````````````````````

更多相关问题可以参照 towardsdatascience_。

.. _towardsdatascience: https://towardsdatascience.com/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2