.. _faq:

Frequently Asked Questions
==========================



Python
---------------------------------------------------------------

.. code-block:: python3

    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    import keras.optimizers as op
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from keras.applications.mobilenet import preprocess_input

    # %%

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
..

