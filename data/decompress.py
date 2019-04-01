# python ./data/decompress.py $TORCH_HOME/ILSVRC2012-TAR/ ./data/data/ILSVRC2012
import os, gc, sys
from pathlib import Path
import multiprocessing


def execute(cmds, idx, num):
  #print ('{:03d} :: {:03d} :: {:03d}'.format(idx, num, len(cmds)))
  for i, cmd in enumerate(cmds):
    if i % num == idx:
      print ('{:03d} :: {:03d} :: {:03d}/{:03d} : {:}'.format(idx, num, i, len(cmds), cmd))
      os.system(cmd)


def command(prefix, cmd):
  #print ('{:}{:}'.format(prefix, cmd))
  #if execute: os.system(cmd)
  return cmd


def main(source, destination, num_process):
  assert source.exists(), '{:} does not exist'.format(source)
  assert (source/'train'  ).exists(), '{:}/train does not exist'.format(source)
  assert (source/'val.tar').exists(), '{:}/val   does not exist'.format(source)
  assert num_process > 0, 'invalid num_process : {:}'.format(num_process)
  source      = source.resolve()
  destination = destination.resolve()
  destination.mkdir(parents=True, exist_ok=True)
  os.system('rm -rf {:}'.format(destination))
  destination.mkdir(parents=True, exist_ok=True)
  (destination/'train').mkdir(parents=True, exist_ok=True)

  subdirs = list( (source / 'train').glob('n*') )
  all_commands = []
  assert len(subdirs) == 1000, 'ILSVRC2012 should contain 1000 classes instead of {:}.'.format( len(subdirs) )
  cmd = command('', 'tar -xf {:} -C {:}'.format(source/'val.tar', destination))
  all_commands.append( cmd )
  for idx, subdir in enumerate(subdirs):
    name = subdir.name
    cmd  = command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), 'tar -xf {:} -C {:}'.format(source/'train'/'{:}'.format(name), destination / 'train'))
    all_commands.append( cmd )
  #print ('Collect all commands done : {:} lines'.format( len(all_commands) ))

  for i, cmd in enumerate(all_commands):
    print(cmd)
  #  os.system(cmd)
  #  print ('{:03d}/{:03d} : {:}'.format(i, len(all_commands), cmd))
  #  gc.collect()

  """
  records = []
  for i in range(num_process):
    process = multiprocessing.Process(target=execute, args=(all_commands, i, num_process))
    process.start()
    records.append(process)
  for process in records:
    process.join()
  """


if __name__ == '__main__':
  assert len(sys.argv) == 4, 'invalid argv : {:}'.format(sys.argv)
  source, destination = Path(sys.argv[1]), Path(sys.argv[2])
  num_process = int(sys.argv[3])
  main(source, destination, num_process)
