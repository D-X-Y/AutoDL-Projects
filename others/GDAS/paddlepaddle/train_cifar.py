import os, sys, numpy as np, argparse
from pathlib import Path
import paddle.fluid as fluid
import math, time, paddle
import paddle.fluid.layers.ops as ops
#from tb_paddle import SummaryWriter

lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from models import resnet_cifar, NASCifarNet, Networks
from utils  import AverageMeter, time_for_file, time_string, convert_secs2time
from utils  import reader_creator


def inference_program(model_name, num_class):
  # The image is 32 * 32 with RGB representation.
  data_shape = [3, 32, 32]
  images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

  if model_name == 'ResNet20':
    predict = resnet_cifar(images,  20, num_class)
  elif model_name == 'ResNet32':
    predict = resnet_cifar(images,  32, num_class)
  elif model_name == 'ResNet110':
    predict = resnet_cifar(images, 110, num_class)
  else:
    predict = NASCifarNet(images, 36, 6, 3, num_class, Networks[model_name], True)
  return predict


def train_program(predict):
  label   = fluid.layers.data(name='label', shape=[1], dtype='int64')
  if isinstance(predict, (list, tuple)):
    predict, aux_predict = predict
    x_losses   = fluid.layers.cross_entropy(input=predict, label=label)
    aux_losses = fluid.layers.cross_entropy(input=aux_predict, label=label)
    x_loss     = fluid.layers.mean(x_losses)
    aux_loss   = fluid.layers.mean(aux_losses)
    loss = x_loss + aux_loss * 0.4
    accuracy = fluid.layers.accuracy(input=predict, label=label)
  else:
    losses  = fluid.layers.cross_entropy(input=predict, label=label)
    loss    = fluid.layers.mean(losses)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
  return [loss, accuracy]


# For training test cost
def evaluation(program, reader, fetch_list, place):
  feed_var_list = [program.global_block().var('pixel'), program.global_block().var('label')]
  feeder_test   = fluid.DataFeeder(feed_list=feed_var_list, place=place)
  test_exe      = fluid.Executor(place)
  losses, accuracies = AverageMeter(), AverageMeter()
  for tid, test_data in enumerate(reader()):
    loss, acc = test_exe.run(program=program, feed=feeder_test.feed(test_data), fetch_list=fetch_list)
    losses.update(float(loss), len(test_data))
    accuracies.update(float(acc)*100, len(test_data))
  return losses.avg, accuracies.avg


def cosine_decay_with_warmup(learning_rate, step_each_epoch, epochs=120):
  """Applies cosine decay to the learning rate.
  lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
  decrease lr for every mini-batch and start with warmup.
  """
  from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
  from paddle.fluid.initializer import init_on_cpu
  global_step = _decay_step_counter()
  lr = fluid.layers.tensor.create_global_var(
      shape=[1],
      value=0.0,
      dtype='float32',
      persistable=True,
      name="learning_rate")

  warmup_epoch = fluid.layers.fill_constant(
      shape=[1], dtype='float32', value=float(5), force_cpu=True)

  with init_on_cpu():
    epoch = ops.floor(global_step / step_each_epoch)
    with fluid.layers.control_flow.Switch() as switch:
      with switch.case(epoch < warmup_epoch):
        decayed_lr = learning_rate * (global_step / (step_each_epoch * warmup_epoch))
        fluid.layers.tensor.assign(input=decayed_lr, output=lr)
      with switch.default():
        decayed_lr = learning_rate * \
          (ops.cos((global_step - warmup_epoch * step_each_epoch) * (math.pi / (epochs * step_each_epoch))) + 1)/2
        fluid.layers.tensor.assign(input=decayed_lr, output=lr)
  return lr


def main(xargs):

  save_dir = Path(xargs.log_dir) / time_for_file()
  save_dir.mkdir(parents=True, exist_ok=True)
  
  print ('save dir : {:}'.format(save_dir))
  print ('xargs : {:}'.format(xargs))

  if xargs.dataset == 'cifar-10':
    train_data = reader_creator(xargs.data_path, 'data_batch', True , False)
    test__data = reader_creator(xargs.data_path, 'test_batch', False, False)
    class_num  = 10
    print ('create cifar-10  dataset')
  elif xargs.dataset == 'cifar-100':
    train_data = reader_creator(xargs.data_path, 'train', True , False)
    test__data = reader_creator(xargs.data_path, 'test' , False, False)
    class_num  = 100
    print ('create cifar-100 dataset')
  else:
    raise ValueError('invalid dataset : {:}'.format(xargs.dataset))
  
  train_reader = paddle.batch(
    paddle.reader.shuffle(train_data, buf_size=5000),
    batch_size=xargs.batch_size)

  # Reader for testing. A separated data set for testing.
  test_reader = paddle.batch(test__data, batch_size=xargs.batch_size)

  place = fluid.CUDAPlace(0)

  main_program = fluid.default_main_program()
  star_program = fluid.default_startup_program()

  # programs
  predict      = inference_program(xargs.model_name, class_num)
  [loss, accuracy] = train_program(predict)
  print ('training program setup done')
  test_program = main_program.clone(for_test=True)
  print ('testing  program setup done')

  #infer_writer = SummaryWriter( str(save_dir / 'infer') )
  #infer_writer.add_paddle_graph(fluid_program=fluid.default_main_program(), verbose=True)
  #infer_writer.close()
  #print(test_program.to_string(True))

  #learning_rate = fluid.layers.cosine_decay(learning_rate=xargs.lr, step_each_epoch=xargs.step_each_epoch, epochs=xargs.epochs)
  #learning_rate = fluid.layers.cosine_decay(learning_rate=0.1, step_each_epoch=196, epochs=300)
  learning_rate = cosine_decay_with_warmup(xargs.lr, xargs.step_each_epoch, xargs.epochs)
  optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005),
            use_nesterov=True)
  optimizer.minimize( loss )

  exe = fluid.Executor(place)

  feed_var_list_loop = [main_program.global_block().var('pixel'), main_program.global_block().var('label')]
  feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
  exe.run(star_program)

  start_time, epoch_time = time.time(), AverageMeter()
  for iepoch in range(xargs.epochs):
    losses, accuracies, steps = AverageMeter(), AverageMeter(), 0
    for step_id, train_data in enumerate(train_reader()):
      tloss, tacc, xlr = exe.run(main_program, feed=feeder.feed(train_data), fetch_list=[loss, accuracy, learning_rate])
      tloss, tacc, xlr = float(tloss), float(tacc) * 100, float(xlr)
      steps += 1
      losses.update(tloss, len(train_data))
      accuracies.update(tacc, len(train_data))
      if step_id % 100 == 0:
        print('{:} [{:03d}/{:03d}] [{:03d}] lr = {:.7f}, loss = {:.4f} ({:.4f}), accuracy = {:.2f} ({:.2f}), error={:.2f}'.format(time_string(), iepoch, xargs.epochs, step_id, xlr, tloss, losses.avg, tacc, accuracies.avg, 100-accuracies.avg))
    test_loss, test_acc = evaluation(test_program, test_reader, [loss, accuracy], place)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (xargs.epochs-iepoch), True) )
    print('{:}x[{:03d}/{:03d}] {:} train-loss = {:.4f}, train-accuracy = {:.2f}, test-loss = {:.4f}, test-accuracy = {:.2f} test-error = {:.2f} [{:} steps per epoch]\n'.format(time_string(), iepoch, xargs.epochs, need_time, losses.avg, accuracies.avg, test_loss, test_acc, 100-test_acc, steps))
    if isinstance(predict, list):
      fluid.io.save_inference_model(str(save_dir / 'inference_model'), ["pixel"],   predict, exe)
    else:
      fluid.io.save_inference_model(str(save_dir / 'inference_model'), ["pixel"], [predict], exe)
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  print('finish training and evaluation with {:} epochs in {:}'.format(xargs.epochs, convert_secs2time(epoch_time.sum, True)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log_dir' ,       type=str,                   help='Save dir.')
  parser.add_argument('--dataset',        type=str,                   help='The dataset name.')
  parser.add_argument('--data_path',      type=str,                   help='The dataset path.')
  parser.add_argument('--model_name',     type=str,                   help='The model name.')
  parser.add_argument('--lr',             type=float,                 help='The learning rate.')
  parser.add_argument('--batch_size',     type=int,                   help='The batch size.')
  parser.add_argument('--step_each_epoch',type=int,                   help='The batch size.')
  parser.add_argument('--epochs'    ,     type=int,                   help='The total training epochs.')
  args = parser.parse_args()
  main(args)
