import numpy as np
import tensorflow as tf
import pdb

FLAGS = tf.app.flags.FLAGS
global ctr
ctr = 0

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features):
        self.name = name
        self.outputs=[features]

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        
        return '%s_L%03d' % (self.name, layer+1)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, training=True):
        """Adds a batch normalization layer to this model.
        See ArXiv 1502.03167v3 for details."""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.layers.batch_normalization(self.get_output(), training=training, name=self._get_layer_str()+"bN")
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str()):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('weight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str()):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self
    
    def add_swish(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.swish(self.get_output())

        self.outputs.append(out)
        return self   
    
    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self
     
    
    def merge(self, growth_rate, isTraining, name):
        
        self.add_batch_norm(training=isTraining)
        self.add_relu()
        self.add_conv2d(growth_rate, mapsize=3, stride=1)
            
        return self

    def add_block_unit(self, growth_rate,isTraining, name):
        
        self.add_batch_norm(training=isTraining)
        self.add_relu()
        self.add_conv2d(4*growth_rate, mapsize=3, stride=1)
        self.merge(growth_rate,isTraining, name)
        
        return self
		
    def add_transition_layer(self, isTraining, theta, name):
        in_x = self.get_output()
        channels = int(in_x.get_shape()[-1])
        self.add_batch_norm(training=isTraining)
        # ~ self.add_relu()
        self.add_conv2d(int(channels*theta), mapsize=3, stride=1)
#         self.add_conv2d(next_layers, mapsize=3, stride=1)
        # ~ x = tf.nn.avg_pool(value=self.get_output(), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'avg_pool_2x2')
        # ~ self.outputs.append(x)
        return self
		
    def add_dense_block(self, growth_rate,isTraining, name):
        in_x = self.get_output()
        channels = int(in_x.get_shape()[-1])
        print("Input channel is %d and want to have %d channels" % (channels, growth_rate))
        with tf.name_scope("block1") as scope:
            self.add_block_unit(growth_rate, isTraining, name+"block1")#NOTE:64 is 2k, here k = 32, 128 is 4k, output is k = 32
            x = tf.concat(values=[in_x, self.get_output()], axis=3, name=name+'stack0')# 96
            self.outputs.append(x)
            print("Denseblock1 with channels",(x.get_shape().as_list()))

        with tf.name_scope("block2") as scope:
            self.add_block_unit(growth_rate, isTraining, name+"block2")
            x = tf.concat(values=[x, self.get_output()], axis=3, name=name+'stack1')
            self.outputs.append(x)
            print("Denseblock2 with channels",(x.get_shape().as_list()))

        with tf.name_scope("block3") as scope:
            self.add_block_unit(growth_rate, isTraining, name+"block3")
            x = tf.concat(values=[x, self.get_output()], axis=3, name=name+'stack2')
            self.outputs.append(x)
            print("Denseblock3 with channels",(x.get_shape().as_list()))

#         with tf.name_scope("block4") as scope:
#             self.add_block_unit(16, isTraining, name+"block4")
#             x = tf.concat(values=[x, self.get_output()], axis=3, name=name+'stack3')
#             self.outputs.append(x)
#             print("Denseblock4 with channels",(x.get_shape().as_list()))

        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3, training=True):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm(training=training)
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False, training=True):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm(training=training)
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_batch_norm(training=training)
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm(training=training)
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            #~ assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def _discriminator_model(sess, features, disc_input, training=False):
    # Fully convolutional model
    mapsize = 3
    layers  = [64, 128, 256, 384]
    

    old_vars = tf.global_variables()

    model = Model('DIS', 2*disc_input - 1)
    #~ pdb.set_trace()

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm(training=training)
        model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm(training=training)
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm(training=training)
    model.add_relu()

    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()

    new_vars  = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), disc_vars

def _generator_model(sess, features, labels, channels, training=True):
    # Upside-down all-convolutional resnet

    mapsize = 3
    if FLAGS.init_layer_size==512:
        res_units  = [512, 384, 256]
        res_units  = [512-48*2, 384-48*3, 256]
    elif FLAGS.init_layer_size==256:
        res_units  = [256, 128, 96]
    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN', features)
#     nums = [2,3,4]
    nums = [2,3,4]
    
    for ru in range(len(res_units)-1):
        
        #=======Comment if you want to use resNet===========
        nunits  = res_units[ru]
        model.add_conv2d(nunits)
        model.add_relu()
        #=======Comment if you want to use resNet===========
        for j in range(nums[ru]):
            '''
            Redisual Network backbone
            '''
            # ~ model.add_residual_block(nunits, mapsize=mapsize, training=training)  #Uncomment if you want to use resnet
            '''
            Densely Connected Network backbone
            '''
            model.add_dense_block(16, training, "denb%d"%(j))
            model.add_transition_layer(training, 1, "transb%d"%(j))
            
        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        model.add_upscale()
        
        model.add_batch_norm(training=training)
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

    # Finalization a la "all convolutional net"
    feat = model.get_output()
    my_shape = feat.get_shape().as_list()
    my_dense_num = my_shape[1]*my_shape[2]*my_shape[3]
    feat = tf.reshape(feat, [-1, my_dense_num])
    w1 = tf.get_variable("finalW", [my_dense_num, 128], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("finalB", [128], initializer=tf.truncated_normal_initializer())
    feat = tf.nn.softmax(tf.matmul(feat, w1) + b1)
    
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    model.add_batch_norm(training=training)
    model.add_relu()
    
    
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    model.add_relu()
    

    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_sigmoid() #Follows by WGAN
    
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars, feat

def create_model(sess, features, labels, features2, labels2, training):
    # Generator
    rows      =     int(features.get_shape()[1])
    cols      =      int(features.get_shape()[2])
    channels  = int(features.get_shape()[3])

    gene_minput =  tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels])
    gene_minput2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels])

    # TBD: Is there a better way to instance the generator?
    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list, feat1 = _generator_model(sess, features, labels, channels,training=training)
        scope.reuse_variables()
        gene_output2, gene_var_list2 , feat2= _generator_model(sess, features2, labels2, channels, training=training)
        scope.reuse_variables()
        gene_moutput, _, _= _generator_model(sess, gene_minput, labels, channels, training=training)
        scope.reuse_variables()
        gene_moutput2, _, _ = _generator_model(sess, gene_minput2, labels2, channels, training=training)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')
    disc_real_input2 = tf.identity(labels2, name='disc_real_input')

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = _discriminator_model(sess, features, disc_real_input,training=training)
        scope.reuse_variables()
        disc_real_output2, disc_var_list2 = _discriminator_model(sess, features2, disc_real_input2, training=training)
        scope.reuse_variables()
        disc_fake_output, _ = _discriminator_model(sess, features, gene_output, training=training)
        scope.reuse_variables()
        disc_fake_output2, _ = _discriminator_model(sess, features2, gene_output2, training=training)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list, 
            gene_minput2,      gene_moutput2,
            gene_output2,      gene_var_list2,
            disc_real_output2, disc_fake_output2, disc_var_list2,
            feat1, feat2]

def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    ks = images.get_shape().as_list()[3]
    arr = np.zeros([K, K, ks, ks])
    for k in range(ks):
        arr[:,:,k,k] = 1.0/(K*K)

    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')

    return downscaled

def create_generator_loss(disc_output, gene_output, features):
    # I.e. did we fool the discriminator?
    #~ pdb.set_trace()
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_output, labels=tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    # I.e. does the result look like the feature?
    K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    assert K == 2 or K == 4 or K == 8    
    downscaled = _downscale(gene_output, K)

    
    gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - features), name='gene_l1_loss')

    if FLAGS.useRecTerm:
        gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_ce_loss,
                           FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')
    else:
        gene_loss = gene_ce_loss
    
    
    return gene_loss

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
    disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
   
    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
    disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    return disc_real_loss, disc_fake_loss

#============================================================
#===============Optimizer=====================================
#============================================================
def create_optimizers(gene_loss, gene_var_list, disc_loss, disc_var_list, sialoss, sia_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5, 	name='gene_optimizer')

        disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, 	name='disc_optimizer')
        sia_opti = tf.train.AdamOptimizer(learning_rate=learning_rate/5.5, beta1=0.5,  	name='sia_optimizer')

        gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
        disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
        sia_minimize     = sia_opti.minimize(sialoss, var_list=sia_var_list, name='sia_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize, sia_minimize, disc_var_list)
