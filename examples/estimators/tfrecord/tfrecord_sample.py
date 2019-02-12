from skimage import io
import tensorflow as tf
import pandas as pd
import numpy as np

def create_testcase():
    image1 = np.ones((5, 4, 3), np.uint8) * 1
    image2 = np.ones((5, 4, 3), np.uint8) * 2
    image3 = np.ones((5, 4, 3), np.uint8) * 3
    io.imsave("image1.jpg", image1)
    io.imsave("image2.jpg", image2)
    io.imsave("image3.jpg", image3)
    write_file = open('testcase.csv', 'w')
    write_file.write("Name,Class1,Class2,Class3\n")
    write_file.write("image1.jpg,1,0,0\n")
    write_file.write("image2.jpg,0,1,0\n")
    write_file.write("image3.jpg,0,0,1\n")
    write_file.close()

def dataset_output_fn(dataset_list, tfrecords_filename):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    def _preprocessing(image):
        return image

    csv_file = pd.read_csv(dataset_list, dtype=np.str)
    dataset = np.asarray(csv_file)

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for i in range(dataset.shape[0]):
        image = io.imread(dataset[i, 0]).astype(np.float32)
        image = _preprocessing(image)
        label = dataset[i,1:].astype(np.float32)

        image_encode = tf.compat.as_bytes(image.tostring())
        image_shape_encode = tf.compat.as_bytes(np.asarray(image.shape).tostring())

        label_encode = tf.compat.as_bytes(label.tostring())
        label_shape_encode = tf.compat.as_bytes(np.asarray(label.shape).tostring())

        feature = {
            "image": _bytes_feature(image_encode),
            "image/shape":_bytes_feature(image_shape_encode),
            "label": _bytes_feature(label_encode),
            "label/shape":_bytes_feature(label_shape_encode),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

def dataset_input_fn(filenames, num_epochs=1, batch_size=4, shuffle=False):
    def parser(record):
        keys_to_features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/shape": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.string, default_value=""),
            "label/shape": tf.FixedLenFeature((), tf.string, default_value=""),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.decode_raw(parsed["image"], tf.float32)
        image_shape = tf.decode_raw(parsed["image/shape"], tf.int64)
        image = tf.reshape(image, image_shape)

        label = tf.decode_raw(parsed["label"], tf.float32)
        label_shape = tf.decode_raw(parsed["label/shape"], tf.int64)
        label = tf.reshape(label, label_shape)

        return image, label

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    #return dataset

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

create_testcase()
dataset_output_fn("testcase.csv", "sample.tfrecords")
#tfrecords_file = ["sample.tfrecords", "sample.tfrecords"]
sess = tf.Session()
print(sess.run(dataset_input_fn("sample.tfrecords", shuffle=True)))
