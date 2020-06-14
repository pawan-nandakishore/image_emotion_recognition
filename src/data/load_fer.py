
import os 
import tensorflow as tf

IMG_HEIGHT = 48
IMG_WIDTH = 48


def get_label(file_path):
  # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
    label = tf.strings.split(parts[-2])[-1]
    return label 

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])




def process_path(file_path):
    print("file path is {}".format(file_path))
    label = get_label(file_path)
  # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label




def load_fer():
    raw_data = "../data/raw/fer2013"
    train_location = os.path.join(raw_data,'train/')

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    list_ds = tf.data.Dataset.list_files("../data/raw/fer2013/train/*/*", seed=1)
    class_ds = tf.data.Dataset.list_files("../data/raw/fer2013/train/*", seed=1)

    CLASS_NAMES = [str(x).split('/')[-1].strip("'").split(" ")[-1] for x in list(class_ds.as_numpy_iterator())]
  
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    return labeled_ds



BATCH_SIZE = 128
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds
