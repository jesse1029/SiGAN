import srez_demo
import srez_input_y
import srez_model_y
import srez_train

import os.path
import random
import numpy as np
import numpy.random
import pdb
import random as rn

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000, "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset', "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8, "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_float('wei_lab', 1, "Weight for label information")

tf.app.flags.DEFINE_string('run', 'demo', "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', 0.7, "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.5, "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020, "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000, "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False, "Log the device where variables are placed.")

tf.app.flags.DEFINE_bool('isOurTest', True, "Log the device where variables are placed.")


tf.app.flags.DEFINE_integer('num_ID', 32, "How much the labels will be test...")


tf.app.flags.DEFINE_integer('sample_size', 64, "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 1000,"Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0, "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('test_vectors', 16,  """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train', "Output folder where training logs are dumped.")
                           
tf.app.flags.DEFINE_string('training_img_dir', '/home/jess/srez/CASIA/CASIA-WebFace/', "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_string('testing_img_dir', '/home/jess/srez/CASIA/CASIA-WebFace/val.txt', "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 2000,  "Time in minutes to train the model")

label=[]
def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    fn=FLAGS.training_img_dir
    fn2=FLAGS.testing_img_dir


    
    
    #~ filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    #~ filenames = sorted(filenames)
    #~ random.shuffle(filenames)
    #~ filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return FLAGS.training_img_dir


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input_y.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=True)
   

    # TBD: Maybe download dataset here

    # Setup async input queues
    
    train_features, train_labels, ylab = srez_input_y.setup_inputs(sess, FLAGS.training_img_dir+"train.txt", image_size=32, crop_size=128)
    test_features,  test_labels, test_y = srez_input_y.setup_inputs(sess,  FLAGS.training_img_dir+"val.txt", image_size=32, crop_size=128)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model_y.create_model(sess, noisy_train_features, train_labels, ylab)

    gene_loss = srez_model_y.create_generator_loss(disc_fake_output, gene_output, train_features, ylab)
    disc_real_loss, disc_fake_loss = \
                     srez_model_y.create_discriminator_loss(disc_real_output, disc_fake_output, ylab)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize, disc_var_list) = \
            srez_model_y.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data)

def main(argv=None):
    # Training or showing off?

    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()
