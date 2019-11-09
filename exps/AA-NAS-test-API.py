import os, sys, time, queue, torch
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from log_utils import time_string
from models import CellStructure

def get_unique_matrix(archs, consider_zero):
  UniquStrs = [arch.to_unique_str(consider_zero) for arch in archs]
  print ('{:} create unique-string done'.format(time_string()))
  sm_matrix = torch.eye(len(archs)).bool()
  for i, _ in enumerate(UniquStrs):
    for j in range(i):
      sm_matrix[i,j] = sm_matrix[j,i] = UniquStrs[i] == UniquStrs[j]
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
  _, _, unique_num = get_unique_matrix(archs, False)
  print ('{:} There are {:} unique architectures (not considering zero).'.format(time_string(), unique_num))
  _, _, unique_num = get_unique_matrix(archs,  True)
  print ('{:} There are {:} unique architectures (considering zero).'.format(time_string(), unique_num))

if __name__ == '__main__':
  check_unique_arch()
