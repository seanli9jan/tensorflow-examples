from skimage import io
import tensorflow as tf
import numpy as np

def create_testcase():
    image1 = np.ones((5, 4, 3), np.uint8) * 1
    image2 = np.ones((5, 4, 3), np.uint8) * 2
    image3 = np.ones((5, 4, 3), np.uint8) * 3
    io.imsave("image1.jpg", image1)
    io.imsave("image2.jpg", image2)
    io.imsave("image3.jpg", image3)

def dataset_input_fn():
    num_epochs=1

    def get_data():
        filenames = ["image1.jpg", "image2.jpg", "image3.jpg"]
        labels = [1, 2, 3]
        return filenames, labels


    # Use a custom SkImage function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
        image_decoded = io.imread(filename.decode())
        print('image_decoded : ', image_decoded.shape, 'filename : ', filename.decode())
        return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
        image_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    filenames, labels = get_data()

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))

    dataset = dataset.map(_resize_function)
    #dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(2)
    dataset = dataset.repeat(num_epochs)
    #return dataset

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

create_testcase()
sess = tf.Session()
#print(dataset_input_fn())
print(sess.run(dataset_input_fn()))
