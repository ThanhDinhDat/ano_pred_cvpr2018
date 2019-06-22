import tensorflow as tf

import unet
import pix2pix

from flownet2.src.flowlib import flow_to_image
from flownet2.src.flownet_sd.flownet_sd import FlowNetSD  # Ok
from flownet2.src.training_schedules import LONG_SCHEDULE
from flownet2.src.net import Mode


slim = tf.contrib.slim


def generator(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    # return unet.unet(inputs, layers, features_root, filter_size, pool_size, output_channel)
    return srgan_generator(inputs, layers, features_root, filter_size, pool_size, output_channel)


def discriminator(inputs, num_filers=(128, 256, 512, 512)):
    logits, end_points = pix2pix.pix2pix_discriminator(inputs, num_filers)
    return logits, end_points['predictions']

def srgan_generator(inputs, layers, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, True)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, True)
            net = net + inputs

        return net


    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, True)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, output_channel, 1, scope='conv')

    return net

# Definition of the discriminator
def srgan_discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net

def flownet(input_a, input_b, height, width, reuse=None):
    net = FlowNetSD(mode=Mode.TEST)
    # train preds flow
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    # input size is 384 x 512
    input_a = tf.image.resize_images(input_a, [height, width])
    input_b = tf.image.resize_images(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    return flows['flow']


def initialize_flownet(sess, checkpoint):
    flownet_vars = slim.get_variables_to_restore(include=['FlowNetSD'])
    flownet_saver = tf.train.Saver(flownet_vars)
    print('FlownetSD restore from {}!'.format(checkpoint))
    flownet_saver.restore(sess, checkpoint)
