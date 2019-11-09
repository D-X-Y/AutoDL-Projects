import os, sys, time


class PrintLogger(object):
  
  def __init__(self):
    """Create a summary writer logging to log_dir."""
    self.name = 'PrintLogger'

  def log(self, string):
    print (string)

  def close(self):
    print ('-'*30 + ' close printer ' + '-'*30)
