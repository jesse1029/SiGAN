import tensorflow as tf
import pdb
import numpy as np
FLAGS = tf.app.flags.FLAGS

def read_labeled_image_list(image_list_file, isSkip=True):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    content = f.readlines()
    glen1 = len(content)
    pair1=[]
    pair2=[]
    labels = []
    st=2
    it=25
    if not isSkip:
        st=0
        it=1

    for i in range(st,glen1, it):
        line = content[i]
        fn1, fn2, label = line[:-1].split(' ')
        fn1 = FLAGS.training_img_dir+fn1
        fn2 = FLAGS.training_img_dir+fn2
        pair1.append(fn1)
        pair2.append(fn2)
        #labels.append(int(label)/10574.0)
        labels.append(int(label))
    return pair1, pair2, np.asarray(labels)
    
    
def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    
    """
    #~ label = tf.one_hot(input_queue[1], 10575)
    label= input_queue[2]
    fn =     input_queue[0]
    fn2 =   input_queue[1]
    #label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3, name="dataset_image")
    file_contents = tf.read_file(input_queue[1])
    example2 = tf.image.decode_jpeg(file_contents, channels=3, name="dataset2_image")
    
    return example, example2, label, fn, fn2
    
def setup_inputs(sess, filenames, image_size=None, capacity_factor=3, crop_size=128, isTest=False, batch_size=None):
    if image_size is None:
        image_size = FLAGS.sample_size
    if batch_size is None:
        batch_size = FLAGS.batch_size
    # Read each JPEG file
    image_list, image_list2, label_list = read_labeled_image_list(filenames, not isTest)
    glen1 = len(label_list)

    images = tf.cast(image_list, tf.string)
    images2 = tf.cast(image_list2, tf.string)
    labels = tf.cast(label_list, tf.int32)


    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, images2, labels],
                                            shuffle= (not isTest))

    image, image2, y, fn, fn2 = read_images_from_disk(input_queue)
    #image = tf.image.adjust_gamma(image, gamma=0.8, gain=1)
    #image = tf.image.per_image_standardization(image)
    
    channels = 3
    image.set_shape([None, None, channels])
    image2.set_shape([None, None, channels])
    
    
    
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
        image2 = tf.image.random_flip_left_right(image2)
        image2 = tf.image.random_saturation(image2, .95, 1.05)
        image2 = tf.image.random_brightness(image2, .05)
        image2 = tf.image.random_contrast(image2, .95, 1.05)
   

	#===========for original dataset===========
    wiggle = 0
    off_x, off_y = 51-wiggle, 92-wiggle
    #crop_size = 32
    #~ crop_size_plus = crop_size + 2*wiggle
    #=============for caisa====================
    off_x, off_y = 60,74
    #~ off_x, off_y = 18,27 # for LFW
    crop_size_plus = crop_size
#     if isTest is False:
#         image = tf.image.crop_to_bounding_box(image, off_y-2, off_x-2, crop_size_plus+4, crop_size_plus+4)
#         image2 = tf.image.crop_to_bounding_box(image2, off_y-2, off_x-2, crop_size_plus+4, crop_size_plus+4)
#         image = tf.random_crop(image, [crop_size_plus,crop_size_plus,3])
#         image2 = tf.random_crop(image2, [crop_size_plus,crop_size_plus,3])
#     else:
    image =  tf.image.crop_to_bounding_box(image,  off_y, off_x, crop_size_plus, crop_size_plus)
    image2 = tf.image.crop_to_bounding_box(image2, off_y, off_x, crop_size_plus, crop_size_plus)
    #pdb.set_trace()
    #~ image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image2 = tf.reshape(image2, [1, crop_size, crop_size, 3])
   

    image =   tf.cast(image, tf.float32)/255
    image2 = tf.cast(image2, tf.float32)/255
   
    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])
        image2 = tf.image.resize_area(image2, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    K = 4
    down_size = image_size//K
   
    downsampled = tf.image.resize_area(image, [down_size, down_size])
    downsampled2 = tf.image.resize_area(image2, [down_size, down_size])

    lr1 = tf.reshape(downsampled, [down_size, down_size, 3])
    lr2 = tf.reshape(downsampled2, [down_size, down_size, 3])
    hr1   = tf.reshape(image,                   [image_size, image_size, 3])
    hr2   = tf.reshape(image2,                   [image_size, image_size, 3])
    
    # Using asynchronous queues
    #pdb.set_trace()
    
    numthr=4 if not isTest else 1
    lr1, lr2, hr1, hr2, y, fn, fn2 = tf.train.batch([lr1, lr2, hr1, hr2, y,fn, fn2],
										  batch_size=batch_size,
										  num_threads=numthr,
										  capacity = batch_size,
										  name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)

    return lr1, lr2, hr1, hr2, y, fn, fn2, glen1
