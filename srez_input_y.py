import tensorflow as tf
import pdb
FLAGS = tf.app.flags.FLAGS

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = FLAGS.training_img_dir+filename
        filenames.append(filename)
        #labels.append(int(label))
        labels.append(int(label)/10574.0)
        #~ pdb.set_trace()
    return filenames, labels
    
    
def read_images_from_disk(input_queue, size1=192):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    #~ example = tf.image.decode_jpeg(file_contents, channels=3, name="dataset_image")
    example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label,fn
    
def setup_inputs(sess, filenames, image_size=None, capacity_factor=3, crop_size=128, isTest=False):
    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    image_list, label_list = read_labeled_image_list(filenames)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.float32)
    #~ pdb.set_trace()
     # Makes an input queue
    thr1 = 1 if isTest is True else 4
    isShuffle = True if isTest is True else False
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue, size1=250) #160 for LFW 250 for casia

    channels = 3
    image.set_shape([None, None, channels])
    
    
    
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
   

    #=============for caisa====================
    off_x, off_y = 60,74
    # ~ off_x, off_y = 39,52
    crop_size_plus = crop_size
    #off_x, off_y = 41,41 # for LFW
    # ~ image = tf.image.crop_to_bounding_box(image, off_y, off_x, 154, 163)
    image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #pdb.set_trace()
    #~ image = tf.random_crop(image, [crop_size, crop_size, 3])

    # ~ image = tf.reshape(image, [1, 154, 163, 3])
    image = tf.reshape(image, [1, crop_size_plus, crop_size_plus, 3])
   

    image = tf.cast(image, tf.float32)/255
   
    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    K = 4
    down_size = image_size//K
   
    downsampled = tf.image.resize_area(image, [down_size, down_size])

    feature = tf.reshape(downsampled, [down_size, down_size, 3])
    label   = tf.reshape(image,                   [image_size, image_size, 3])
    
    # Using asynchronous queues
    #pdb.set_trace()
    
    
    features, labels,y,fn = tf.train.batch([feature, label, y, fn],
										  batch_size=FLAGS.batch_size,
										  num_threads=thr1,
										  capacity = capacity_factor*FLAGS.batch_size,
										  name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)

    return features, labels, y,fn
