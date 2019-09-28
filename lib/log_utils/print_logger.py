##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import importlib, warnings
import os, sys, time, numpy as np


class PrintLogger(object):
  
  def __init__(self):
    """Create a summary writer logging to log_dir."""
    self.name = 'PrintLogger'

  def log(self, string):
    print (string)

  def close(self):
    print ('-'*30 + ' close printer ' + '-'*30)
