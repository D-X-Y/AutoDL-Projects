import os

class GPUManager():
  queries = ('index', 'gpu_name', 'memory.free', 'memory.used', 'memory.total', 'power.draw', 'power.limit')

  def __init__(self):
    all_gpus = self.query_gpu(False)

  def get_info(self, ctype):
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(ctype)
    lines = os.popen(cmd).readlines()
    lines = [line.strip('\n') for line in lines]
    return lines

  def query_gpu(self, show=True):
    num_gpus = len( self.get_info('index') )
    all_gpus = [ {} for i in range(num_gpus) ]
    for query in self.queries:
      infos = self.get_info(query)
      for idx, info in enumerate(infos):
        all_gpus[idx][query] = info

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
      selected_gpus = []
      for idx, CUDA_VISIBLE_DEVICE in enumerate(CUDA_VISIBLE_DEVICES):
        find = False
        for gpu in all_gpus:
          if gpu['index'] == CUDA_VISIBLE_DEVICE:
            assert find==False, 'Duplicate cuda device index : {}'.format(CUDA_VISIBLE_DEVICE)
            find = True
            selected_gpus.append( gpu.copy() )
            selected_gpus[-1]['index'] = '{}'.format(idx)
        assert find, 'Does not find the device : {}'.format(CUDA_VISIBLE_DEVICE)
      all_gpus = selected_gpus
    
    if show:
      allstrings = ''
      for gpu in all_gpus:
        string = '| '
        for query in self.queries:
          if query.find('memory') == 0: xinfo = '{:>9}'.format(gpu[query])
          else:                         xinfo = gpu[query]
          string = string + query + ' : ' + xinfo + ' | '
        allstrings = allstrings + string + '\n'
      return allstrings
    else:
      return all_gpus

  def select_by_memory(self, numbers=1):
    all_gpus = self.query_gpu(False)
    assert numbers <= len(all_gpus), 'Require {} gpus more than you have'.format(numbers)
    alls = []
    for idx, gpu in enumerate(all_gpus):
      free_memory = gpu['memory.free']
      free_memory = free_memory.split(' ')[0]
      free_memory = int(free_memory)
      index = gpu['index']
      alls.append((free_memory, index))
    alls.sort(reverse = True)
    alls = [ int(alls[i][1]) for i in range(numbers) ]
    return sorted(alls)

"""
if __name__ == '__main__':
  manager = GPUManager()
  manager.query_gpu(True)
  indexes = manager.select_by_memory(3)
  print (indexes)
"""
