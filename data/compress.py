# python ./data/compress.py $TORCH_HOME/ILSVRC2012/ $TORCH_HOME/ILSVRC2012-TAR
import os, sys
from pathlib import Path


def command(prefix, cmd):
  print ('{:}{:}'.format(prefix, cmd))
  os.system(cmd)


def main(source, destination):
  assert source.exists(), '{:} does not exist'.format(source)
  assert (source/'train').exists(), '{:}/train does not exist'.format(source)
  assert (source/'val'  ).exists(), '{:}/val   does not exist'.format(source)
  source      = source.resolve()
  destination = destination.resolve()
  destination.mkdir(parents=True, exist_ok=True)
  os.system('rm -rf {:}'.format(destination))
  destination.mkdir(parents=True, exist_ok=True)
  (destination/'train').mkdir(parents=True, exist_ok=True)

  subdirs = list( (source / 'train').glob('n*') )
  assert len(subdirs) == 1000, 'ILSVRC2012 should contain 1000 classes instead of {:}.'.format( len(subdirs) )
  command('', 'tar -cf {:} -C {:} val'.format(destination/'val.tar', source))
  for idx, subdir in enumerate(subdirs):
    name = subdir.name
    command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), 'tar -cf {:} -C {:} {:}'.format(destination/'train'/'{:}.tar'.format(name), source / 'train', name))


if __name__ == '__main__':
  assert len(sys.argv) == 3, 'invalid argv : {:}'.format(sys.argv)
  source, destination = Path(sys.argv[1]), Path(sys.argv[2])
  main(source, destination)
