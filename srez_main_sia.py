import srez_demo
import srez_input_sia
import srez_model_sia

import os.path
import random
import numpy as np
import numpy.random
import pdb
import random as rn
import os.path
import scipy.misc
import time
import tensorflow as tf
import io

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of samples per batch.")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Output folder where checkpoints are dumped.")
tf.app.flags.DEFINE_integer('checkpoint_period', 1000, "Number of batches in between checkpoints")
tf.app.flags.DEFINE_string('dataset', 'dataset', "Path to the dataset directory.")
tf.app.flags.DEFINE_float('epsilon', 1e-8, "Fuzz term to avoid numerical instability")
tf.app.flags.DEFINE_string('run', 'demo', "Which operation to run. [demo|train|finetune]")
tf.app.flags.DEFINE_float('gene_l1_factor', .90, "Multiplier for generator L1 loss term")
tf.app.flags.DEFINE_float('learning_beta1', 0.5, "Beta1 parameter used for AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate_start', 0.00020, "Starting learning rate used for AdamOptimizer")
tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000, "Number of batches until learning rate is halved")
tf.app.flags.DEFINE_bool('log_device_placement', False, "Log the device where variables are placed.")
tf.app.flags.DEFINE_integer('sample_size', 32, "Image sample size in pixels. Range [64,128]")
tf.app.flags.DEFINE_integer('summary_period', 1000,"Number of batches between summary data dumps")
tf.app.flags.DEFINE_integer('random_seed', 0, "Seed used to initialize rng.")
tf.app.flags.DEFINE_integer('init_layer_size', 512, "Seed used to initialize rng.")
tf.app.flags.DEFINE_integer('test_vectors', 16,  """Number of features to use for testing""")      
tf.app.flags.DEFINE_string('train_dir', 'train', "Output folder where training logs are dumped.")                      
tf.app.flags.DEFINE_string('log_dir', 'logs/', "Output folder where training logs are dumped.")      
tf.app.flags.DEFINE_string('training_img_dir', 'CASIA/CASIA-WebFace/', "Output folder where training logs are dumped.")
tf.app.flags.DEFINE_string('testing_img_dir', 'CASIA/CASIA-WebFace/val.txt', "Output folder where training logs are dumped.")
tf.app.flags.DEFINE_integer('train_time', 10000,  "Time in minutes to train the model")
tf.app.flags.DEFINE_bool('useRecTerm', True,  "Set whether the reconstruction term is used or not")

label=[]

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        if not tf.gfile.Exists(FLAGS.train_dir):
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
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    return sess, summary_writer

def _summarize_progress(sess, feature, label, gene_output, batch, suffix, max_samples=17):

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
    #~ pdb.set_trace()
    image   = tf.concat([nearest, bicubic, clipped, label], 2)
    tf.summary.image("myResults", image)

    imagex = image[0:max_samples,:,:,:]
    image = tf.concat([imagex[i,:,:,:] for i in range(max_samples)], 0)
    image = sess.run(image)
        

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))
    
    

    return imagex

def _save_checkpoint(sess, batch, saver):

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'
    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    
    saver.save(sess, newname)

    print("    Checkpoint saved")


def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=True)
  
    
    train_features, train_features2, train_labels, train_labels2, ylab, fn1, fn2, glen = srez_input_sia.setup_inputs(sess, FLAGS.training_img_dir+"pairwise.txt", image_size=32, crop_size=128)
    test_features, test_features2, test_labels, test_labels2, ytlab, fnt1, fnt2, tlen = srez_input_sia.setup_inputs(sess, FLAGS.training_img_dir+"pairwise-val2.txt", image_size=32, crop_size=128, isTest=True)
    print("Now we have %d training image pairs to be loaded..."%(glen))

   
    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + tf.random_normal(train_features.get_shape(), stddev=noise_level)
    noisy_train_features2 = train_features2 + tf.random_normal(train_features2.get_shape(), stddev=noise_level)

    # Create and initialize model
    #==========Network1==================
    #~ pdb.set_trace()
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
    disc_real_output, disc_fake_output, disc_var_list, 
    gene_minput2, gene_moutput2,
     gene_output2, gene_var_list2,
    disc_real_output2, disc_fake_output2, disc_var_list2,
    feat1, feat2] = srez_model_sia.create_model(sess, noisy_train_features, train_labels, noisy_train_features2, train_labels2, True)

    with tf.name_scope('Generator_loss'):
         gene_loss = srez_model_sia.create_generator_loss(disc_fake_output, gene_output, train_features)
         tf.summary.scalar("Generator_loss", gene_loss)
    
    disc_real_loss, disc_fake_loss =   srez_model_sia.create_discriminator_loss(disc_real_output, disc_fake_output)
    with tf.name_scope('Discriminator_loss'):
        disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
        tf.summary.scalar("D_Real_loss", disc_real_loss)
        tf.summary.scalar("D_Fake_loss", disc_fake_loss)
        tf.summary.scalar("Discriminator_loss", disc_loss)
			 
    
    #============Siamese contrastive loss==================
    margin = 0.5
    labels_t = tf.cast(ylab, tf.float32)
    labels_f = tf.cast(1-ylab, tf.float32)         # labels_ = !labels;
    eucd2 = tf.pow(feat1- feat2, 2.0)
    eucd2 = tf.reduce_sum(eucd2, [1])
    eucd = tf.sqrt(eucd2+1e-10, name="eucd")
    C = tf.constant(margin, name="C")
    pos = labels_t * eucd2
    neg = labels_f *tf.pow(tf.maximum(C- eucd, 0), 2)
    losses = pos + neg
    with tf.name_scope('Contrastive_loss'):
        sialoss = tf.reduce_mean(losses, name="Contrastive_loss")
        tf.summary.scalar("Contrastive_loss", sialoss)
    
    #~ gene_loss = tf.add(gene_loss, sialoss*0.1)
    
    (global_step, learning_rate, gene_minimize, disc_minimize, sia_minimize, disc_var_list) =  \
    srez_model_sia.create_optimizers(gene_loss, gene_var_list, disc_loss, disc_var_list, sialoss,gene_var_list)

    # Train model

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0
    #~ pdb.set_trace()
    assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)
    test_feature, test_label = sess.run([test_features, test_labels])
    ops =  [disc_minimize, disc_real_loss, disc_fake_loss, gene_minimize, gene_loss, sialoss, sia_minimize]
    critic_itrs=4
    cri = glen/100/FLAGS.batch_size
    batch_total = glen/FLAGS.batch_size
    gene_loss2 = disc_real_loss2 = disc_fake_loss2= sia_loss2 = -1.234
    
    while not done:
        batch += 1
        feed_dict = {learning_rate : lrval, gene_minput: test_feature, gene_minput2: test_feature}       
        _, disc_real_loss2, disc_fake_loss2,_, gene_loss2, sia_loss2, _= sess.run(ops, feed_dict=feed_dict)
   
        if batch % 50 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Batch[%d/%d(%3.3f%%)], Epoch[%2d] G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f], Siamese_Loss[%3.3f]' %
                  (batch%batch_total, batch_total, batch/float(batch_total)*100, int(np.floor(batch/float(batch_total))),
                   gene_loss2, disc_real_loss2, disc_fake_loss2, sia_loss2))
        
            current_progress = elapsed / FLAGS.train_time
          
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {gene_minput: test_feature}
            gene_output = sess.run(gene_moutput, feed_dict=feed_dict)
            image = _summarize_progress(sess,test_feature, test_label, gene_output, batch, 'out')
    
            #image_summary_t = tf.summary.image("Train_Images", image)
           # image_summary = sess.run(image_summary_t)
            
            summary_str = sess.run(summaries)
            summary_writer.add_summary(summary_str, batch)
            #summary_writer.add_summary(image_summary)
            
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(sess, batch, saver)

    _save_checkpoint(sess, batch, saver)
    print('Finished training!')

def main(argv=None):
    _train()

if __name__ == '__main__':
  tf.app.run()
