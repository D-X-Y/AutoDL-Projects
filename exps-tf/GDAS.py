# CUDA_VISIBLE_DEVICES=0 python exps-tf/GDAS.py
import os, sys, time, random, argparse
import tensorflow as tf
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

# self-lib
from tf_models import get_cell_based_tiny_net
from tf_optimizers import SGDW, AdamW
from config_utils import dict2config
from log_utils import time_string
from models import CellStructure


def pre_process(image_a, label_a, image_b, label_b):
  def standard_func(image):
    x = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x
  return standard_func(image_a), label_a, standard_func(image_b), label_b


def main(xargs):
  cifar10 = tf.keras.datasets.cifar10

  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

  # Add a channels dimension
  all_indexes = list(range(x_train.shape[0]))
  random.shuffle(all_indexes)
  s_train_idxs, s_valid_idxs = all_indexes[::2], all_indexes[1::2]
  search_train_x, search_train_y = x_train[s_train_idxs], y_train[s_train_idxs]
  search_valid_x, search_valid_y = x_train[s_valid_idxs], y_train[s_valid_idxs]
  #x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]
  
  # Use tf.data
  #train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
  search_ds = tf.data.Dataset.from_tensor_slices((search_train_x, search_train_y, search_valid_x, search_valid_y))
  search_ds = search_ds.map(pre_process).shuffle(1000).batch(64)

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

  # Create an instance of the model
  config = dict2config({'name': 'GDAS',
                        'C'   : xargs.channel, 'N': xargs.num_cells, 'max_nodes': xargs.max_nodes,
                        'num_classes': 10, 'space': 'nas-bench-201', 'affine': True}, None)
  model = get_cell_based_tiny_net(config)
  #import pdb; pdb.set_trace()
  #model.build(((64, 32, 32, 3), (1,)))
  #for x in model.trainable_variables:
  #  print('{:30s} : {:}'.format(x.name, x.shape))
  # Choose optimizer
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  w_optimizer = SGDW(learning_rate=xargs.w_lr, weight_decay=xargs.w_weight_decay, momentum=xargs.w_momentum, nesterov=True)
  a_optimizer = AdamW(learning_rate=xargs.arch_learning_rate, weight_decay=xargs.arch_weight_decay, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
  #w_optimizer = tf.keras.optimizers.SGD(learning_rate=0.025, momentum=0.9, nesterov=True)
  #a_optimizer = tf.keras.optimizers.AdamW(learning_rate=xargs.arch_learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-07)
  ####
  # metrics
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  valid_loss = tf.keras.metrics.Mean(name='valid_loss')
  valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  
  @tf.function
  def search_step(train_images, train_labels, valid_images, valid_labels, tf_tau):
    # optimize weights
    with tf.GradientTape() as tape:
      predictions = model(train_images, tf_tau, True)
      w_loss = loss_object(train_labels, predictions)
    net_w_param = model.get_weights()
    gradients = tape.gradient(w_loss, net_w_param)
    w_optimizer.apply_gradients(zip(gradients, net_w_param))
    train_loss(w_loss)
    train_accuracy(train_labels, predictions)
    # optimize alphas
    with tf.GradientTape() as tape:
      predictions = model(valid_images, tf_tau, True)
      a_loss = loss_object(valid_labels, predictions)
    net_a_param = model.get_alphas()
    gradients = tape.gradient(a_loss, net_a_param)
    a_optimizer.apply_gradients(zip(gradients, net_a_param))
    valid_loss(a_loss)
    valid_accuracy(valid_labels, predictions)

  # TEST
  @tf.function
  def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  print('{:} start searching with {:} epochs ({:} batches per epoch).'.format(time_string(), xargs.epochs, tf.data.experimental.cardinality(search_ds).numpy()))

  for epoch in range(xargs.epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states() ; train_accuracy.reset_states()
    test_loss.reset_states()  ; test_accuracy.reset_states()
    cur_tau = xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (xargs.epochs-1)
    tf_tau  = tf.cast(cur_tau, dtype=tf.float32, name='tau')

    for trn_imgs, trn_labels, val_imgs, val_labels in search_ds:
      search_step(trn_imgs, trn_labels, val_imgs, val_labels, tf_tau)
    genotype = model.genotype()
    genotype = CellStructure(genotype)

    #for test_images, test_labels in test_ds:
    #  test_step(test_images, test_labels)

    template = '{:} Epoch {:03d}/{:03d}, Train-Loss: {:.3f}, Train-Accuracy: {:.2f}%, Valid-Loss: {:.3f}, Valid-Accuracy: {:.2f}% | tau={:.3f}'
    print(template.format(time_string(), epoch+1, xargs.epochs,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          valid_loss.result(),
                          valid_accuracy.result()*100,
                          cur_tau))
    print('{:} genotype : {:}\n{:}\n'.format(time_string(), genotype, model.get_np_alphas()))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='NAS-Bench-201', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # training details
  parser.add_argument('--epochs'            , type=int  ,   default= 250  ,   help='')
  parser.add_argument('--tau_max'           , type=float,   default= 10   ,   help='')
  parser.add_argument('--tau_min'           , type=float,   default= 0.1  ,   help='')
  parser.add_argument('--w_lr'              , type=float,   default= 0.025,   help='')
  parser.add_argument('--w_weight_decay'    , type=float,   default=0.0005,   help='')
  parser.add_argument('--w_momentum'        , type=float,   default= 0.9  ,   help='')
  parser.add_argument('--arch_learning_rate', type=float,   default=0.0003,   help='')
  parser.add_argument('--arch_weight_decay' , type=float,   default=0.001,    help='')
  # marco structure
  parser.add_argument('--channel'           , type=int  ,   default=16,       help='')
  parser.add_argument('--num_cells'         , type=int  ,   default= 5,       help='')
  parser.add_argument('--max_nodes'         , type=int  ,   default= 4,       help='')
  args = parser.parse_args()
  main( args )
