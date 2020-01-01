##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import sys, time, torch, random, argparse
from PIL     import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy    import deepcopy
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, obtain_cls_kd_args as obtain_args
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from procedures   import get_optim_scheduler, get_procedures
from datasets     import get_datasets
from models       import obtain_model, load_net_from_checkpoint
from utils        import get_model_infos
from log_utils    import AverageMeter, time_string, convert_secs2time


def main(args):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  #torch.backends.cudnn.deterministic = True
  torch.set_num_threads( args.workers )
  
  prepare_seed(args.rand_seed)
  logger = prepare_logger(args)
  
  train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True , num_workers=args.workers, pin_memory=True)
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
  # get configures
  model_config = load_config(args.model_config, {'class_num': class_num}, logger)
  optim_config = load_config(args.optim_config,
                                {'class_num': class_num, 'KD_alpha': args.KD_alpha, 'KD_temperature': args.KD_temperature},
                                logger)

  # load checkpoint
  teacher_base = load_net_from_checkpoint(args.KD_checkpoint)
  teacher      = torch.nn.DataParallel(teacher_base).cuda()

  base_model   = obtain_model(model_config)
  flop, param  = get_model_infos(base_model, xshape)
  logger.log('Student ====>>>>:\n{:}'.format(base_model))
  logger.log('Teacher ====>>>>:\n{:}'.format(teacher_base))
  logger.log('model information : {:}'.format(base_model.get_message()))
  logger.log('-'*50)
  logger.log('Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G'.format(param, flop, flop/1e3))
  logger.log('-'*50)
  logger.log('train_data : {:}'.format(train_data))
  logger.log('valid_data : {:}'.format(valid_data))
  optimizer, scheduler, criterion = get_optim_scheduler(base_model.parameters(), optim_config)
  logger.log('optimizer  : {:}'.format(optimizer))
  logger.log('scheduler  : {:}'.format(scheduler))
  logger.log('criterion  : {:}'.format(criterion))
  
  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(base_model).cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch'] + 1
    checkpoint  = torch.load(last_info['last_checkpoint'])
    base_model.load_state_dict( checkpoint['base-model'] )
    scheduler.load_state_dict ( checkpoint['scheduler'] )
    optimizer.load_state_dict ( checkpoint['optimizer'] )
    valid_accuracies = checkpoint['valid_accuracies']
    max_bytes        = checkpoint['max_bytes']
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  elif args.resume is not None:
    assert Path(args.resume).exists(), 'Can not find the resume file : {:}'.format(args.resume)
    checkpoint  = torch.load( args.resume )
    start_epoch = checkpoint['epoch'] + 1
    base_model.load_state_dict( checkpoint['base-model'] )
    scheduler.load_state_dict ( checkpoint['scheduler'] )
    optimizer.load_state_dict ( checkpoint['optimizer'] )
    valid_accuracies = checkpoint['valid_accuracies']
    max_bytes        = checkpoint['max_bytes']
    logger.log("=> loading checkpoint from '{:}' start with {:}-th epoch.".format(args.resume, start_epoch))
  elif args.init_model is not None:
    assert Path(args.init_model).exists(), 'Can not find the initialization file : {:}'.format(args.init_model)
    checkpoint  = torch.load( args.init_model )
    base_model.load_state_dict( checkpoint['base-model'] )
    start_epoch, valid_accuracies, max_bytes = 0, {'best': -1}, {}
    logger.log('=> initialize the model from {:}'.format( args.init_model ))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, max_bytes = 0, {'best': -1}, {}

  train_func, valid_func = get_procedures(args.procedure)
  
  total_epoch = optim_config.epochs + optim_config.warmup
  # Main Training and Evaluation Loop
  start_time  = time.time()
  epoch_time  = AverageMeter()
  for epoch in range(start_epoch, total_epoch):
    scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (total_epoch-epoch), True) )
    epoch_str = 'epoch={:03d}/{:03d}'.format(epoch, total_epoch)
    LRs       = scheduler.get_lr()
    find_best = False

    logger.log('\n***{:s}*** start {:s} {:s}, LR=[{:.6f} ~ {:.6f}], scheduler={:}'.format(time_string(), epoch_str, need_time, min(LRs), max(LRs), scheduler))
    
    # train for one epoch
    train_loss, train_acc1, train_acc5 = train_func(train_loader, teacher, network, criterion, scheduler, optimizer, optim_config, epoch_str, args.print_freq, logger)
    # log the results    
    logger.log('***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}'.format(time_string(), epoch_str, train_loss, train_acc1, train_acc5))

    # evaluate the performance
    if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
      logger.log('-'*150)
      valid_loss, valid_acc1, valid_acc5 = valid_func(valid_loader, teacher, network, criterion, optim_config, epoch_str, args.print_freq_eval, logger)
      valid_accuracies[epoch] = valid_acc1
      logger.log('***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}'.format(time_string(), epoch_str, valid_loss, valid_acc1, valid_acc5, valid_accuracies['best'], 100-valid_accuracies['best']))
      if valid_acc1 > valid_accuracies['best']:
        valid_accuracies['best'] = valid_acc1
        find_best                = True
        logger.log('Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, acc@5={:.2f}, error@1={:.2f}, error@5={:.2f}, save into {:}.'.format(epoch, valid_acc1, valid_acc5, 100-valid_acc1, 100-valid_acc5, model_best_path))
      num_bytes = torch.cuda.max_memory_cached( next(network.parameters()).device ) * 1.0
      logger.log('[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]'.format(next(network.parameters()).device, int(num_bytes), num_bytes / 1e3, num_bytes / 1e6, num_bytes / 1e9))
      max_bytes[epoch] = num_bytes
    if epoch % 10 == 0: torch.cuda.empty_cache()

    # save checkpoint
    save_path = save_checkpoint({
          'epoch'        : epoch,
          'args'         : deepcopy(args),
          'max_bytes'    : deepcopy(max_bytes),
          'FLOP'         : flop,
          'PARAM'        : param,
          'valid_accuracies': deepcopy(valid_accuracies),
          'model-config' : model_config._asdict(),
          'optim-config' : optim_config._asdict(),
          'base-model'   : base_model.state_dict(),
          'scheduler'    : scheduler.state_dict(),
          'optimizer'    : optimizer.state_dict(),
          }, model_base_path, logger)
    if find_best: copy_checkpoint(model_base_path, model_best_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*200)
  logger.log('||| Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G'.format(param, flop, flop/1e3))
  logger.log('Finish training/validation in {:} with Max-GPU-Memory of {:.2f} MB, and save final checkpoint into {:}'.format(convert_secs2time(epoch_time.sum, True), max(v for k, v in max_bytes.items()) / 1e6, logger.path('info')))
  logger.log('-'*200 + '\n')
  logger.close()


if __name__ == '__main__':
  args = obtain_args()
  main(args)
