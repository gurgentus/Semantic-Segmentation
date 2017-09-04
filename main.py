import os.path
import tensorflow as tf
import helper
import warnings
import shutil
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    return sess.graph.get_tensor_by_name(vgg_input_tensor_name), \
            sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name), \
            sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name), \
            sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name), \
            sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # picture of architecture: page 4 of
    # https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    # Resources:
    # VGG16 Visualization: https://www.cs.toronto.edu/~frossard/post/vgg16/
    # VGG16 Keras Implementation: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    # make sure the shapes are the same!
    input = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1))
    input = tf.layers.conv2d_transpose(input, num_classes, 1, strides=(1,1))
    input = tf.layers.conv2d_transpose(input, 512, 4, strides=(2, 2), padding='SAME')
    input = tf.add(input, vgg_layer4_out)
    input = tf.layers.conv2d_transpose(input, 256, 4, strides=(2, 2), padding='SAME')
    input = tf.add(input, vgg_layer3_out)
    output = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8), padding='SAME')

    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    loss_operation = tf.reduce_mean(cross_entropy_loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_operation)

    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    merged = tf.summary.merge_all()
    print("start training")
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print("epoch")
        for batch_x, batch_y in get_batches_fn(batch_size):
            _, loss_val = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_x, keep_prob: 1, correct_label: batch_y})
        print ('loss = ', loss_val)

    pass

data_dir = './data'
data_folder = 'data_road/testing'
runs_dir = './runs'

def run():
    num_classes = 2
    image_shape = (160, 576)


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = 0.001
    keep_prob = 1
    EPOCHS = 15
    BATCH_SIZE = 2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], 2))


        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, vgg_input,
             correct_label, vgg_keep_prob, learning_rate)

        # Save the variables to disk.
        save_path = saver.save(sess, "models/model.ckpt")
        print("Model saved in file: %s" % save_path)

        print("inference")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

def inference(filename):
    with tf.Session() as sess:
      # Restore variables from disk.
      saver.restore(sess, "/tmp/model.ckpt")
      print("Model restored.")

      # Make folder for current run
      output_dir = os.path.join(runs_dir, str(time.time()))
      if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    # image_outputs = gen_test_output(
    #     sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)

    image = scipy.misc.imresize(scipy.misc.imread(filename), image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    image_outputs = os.path.basename(image_file), np.array(street_im)

    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


if __name__ == '__main__':
    run()
