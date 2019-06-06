import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time, pdb

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
    #~ pdb.set_trace()
    image   = tf.concat([nearest, bicubic, clipped, label], 2)

    with tf.name_scope('output_train_results'):
        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(max_samples)], 0)
        image = td.sess.run(image)
        

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))
    return image

def _save_checkpoint(train_data, batch):
    td = train_data

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
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    td.sess.run(tf.global_variables_initializer())


    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0
    #~ pdb.set_trace()
    assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    #~ clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01,0.01)) for  var in td.disc_var_list]
    critic_itrs=0
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        feed_dict = {td.learning_rate : lrval}

        #~ disc_ops = [td.disc_minimize, td.disc_real_loss, td.disc_fake_loss]
        #~ gen_ops = [td.gene_minimize, td.gene_loss]
        
        ops = [td.disc_minimize, td.disc_real_loss, td.disc_fake_loss, td.gene_minimize, td.gene_loss, td.sia_minimize, td.sialoss]
        
        #~ td.sess.run(clip_discriminator_var_op)

        #~ if batch < 25 or batch % 500 == 0:
             #~ critic_itrs = 1
        #~ else:
            #~ critic_itrs =1
#~ 
        #~ for critic_itr in range(critic_itrs):
            #~ _, disc_real_loss, disc_fake_loss = td.sess.run(disc_ops, feed_dict=feed_dict)
            #~ td.sess.run(clip_discriminator_var_op)

        #~ _, gene_loss= td.sess.run(gen_ops, feed_dict=feed_dict)
        _, disc_real_loss, disc_fake_loss, _, gene_loss, _, sia_loss= td.sess.run(ops, feed_dict=feed_dict)
        if batch % 50 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f], Siamese_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed,
                   batch, gene_loss, disc_real_loss, disc_fake_loss, sia_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            summary_str = td.sess.run(summaries)
            td.summary_writer.add_summary(summary_str,batch)
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
            with tf.name_scope('Results'):
                image= _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
                tf.summary.image('Results', image, FLAGS.batch_size)

            summary_str = td.sess.run(summaries)
            td.summary_writer.add_summary(summary_str, batch)
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')
