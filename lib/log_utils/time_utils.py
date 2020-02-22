#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import time, sys
import numpy as np

def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_string_short():
  ISOTIMEFORMAT='%Y%m%d'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_print(string, is_print=True):
  if (is_print):
    print('{} : {}'.format(time_string(), string))

def convert_secs2time(epoch_time, return_str=False):    
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)  
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if return_str:
    str = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return str
  else:
    return need_hour, need_mins, need_secs

def print_log(print_string, log):
  #if isinstance(log, Logger): log.log('{:}'.format(print_string))
  if hasattr(log, 'log'): log.log('{:}'.format(print_string))
  else:
    print("{:}".format(print_string))
    if log is not None:
      log.write('{:}\n'.format(print_string))
      log.flush()
