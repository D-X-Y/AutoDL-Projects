import os, sys, time, queue, torch
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from log_utils  import time_string
from aa_nas_api import AANASBenchAPI, ArchResults
from models     import CellStructure


def get_unique_matrix(archs, consider_zero):
  UniquStrs = [arch.to_unique_str(consider_zero) for arch in archs]
  print ('{:} create unique-string ({:}/{:}) done'.format(time_string(), len(set(UniquStrs)), len(UniquStrs)))
  Unique2Index = dict()
  for index, xstr in enumerate(UniquStrs):
    if xstr not in Unique2Index: Unique2Index[xstr] = list()
    Unique2Index[xstr].append( index )
  sm_matrix = torch.eye(len(archs)).bool()
  for _, xlist in Unique2Index.items():
    for i in xlist:
      for j in xlist:
        sm_matrix[i,j] = True
  unique_ids, unique_num = [-1 for _ in archs], 0
  for i in range(len(unique_ids)):
    if unique_ids[i] > -1: continue
    neighbours = sm_matrix[i].nonzero().view(-1).tolist()
    for nghb in neighbours:
      assert unique_ids[nghb] == -1, 'impossible'
      unique_ids[nghb] = unique_num
    unique_num += 1
  return sm_matrix, unique_ids, unique_num


def check_unique_arch():
  print ('{:} start'.format(time_string()))
  meta_info = torch.load('./output/AA-NAS-BENCH-4/meta-node-4.pth')
  arch_strs = meta_info['archs']
  archs     = [CellStructure.str2structure(arch_str) for arch_str in arch_strs]
  """
  for i, arch in enumerate(archs):
    if not arch.check_valid():
      print('{:05d} {:}'.format(i, arch))
      #start = int(i / 390.) * 390
      #xxend = start + 389
      #print ('/home/dxy/search-configures/output/TINY-NAS-BENCHMARK-4/{:06d}-{:06d}-C16-N5/arch-{:06d}-seed-0888.pth'.format(start, xxend, i))
  """
  print ('There are {:} valid-archs'.format( sum(arch.check_valid() for arch in archs) ))
  sm_matrix, uniqueIDs, unique_num = get_unique_matrix(archs, False)
  save_dir = './output/cell-search-tiny/same-matrix.pth'
  torch.save(sm_matrix, save_dir)
  print ('{:} There are {:} unique architectures (not considering zero).'.format(time_string(), unique_num))
  sm_matrix, uniqueIDs, unique_num = get_unique_matrix(archs,  True)
  print ('{:} There are {:} unique architectures (considering zero).'.format(time_string(), unique_num))


def test_aa_nas_api():
  arch_result = ArchResults.create_from_state_dict('output/AA-NAS-BENCH-4/simplifies/architectures/000002-FULL.pth')
  arch_result.show(True)
  result = arch_result.query('cifar100')
  #xfile = '/home/dxy/search-configures/output/TINY-NAS-BENCHMARK-4/simplifies/C16-N5-final-infos.pth'
  #api   = AANASBenchAPI(xfile)
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  #check_unique_arch()
  test_aa_nas_api()
