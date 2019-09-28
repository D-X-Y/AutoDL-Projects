# python ./data/compress.py $TORCH_HOME/ILSVRC2012/ $TORCH_HOME/ILSVRC2012-TAR tar
# python ./data/compress.py $TORCH_HOME/ILSVRC2012/ $TORCH_HOME/ILSVRC2012-ZIP zip
import os, sys
from pathlib import Path


def command(prefix, cmd):
  print ('{:}{:}'.format(prefix, cmd))
  os.system(cmd)


def main(source, destination, xtype):
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
  if xtype == 'tar'  : command('', 'tar -cf {:} -C {:} val'.format(destination/'val.tar', source))
  elif xtype == 'zip': command('', '(cd {:} ; zip -r {:} val)'.format(source, destination/'val.zip'))
  else: raise ValueError('invalid compress type : {:}'.format(xtype))
  for idx, subdir in enumerate(subdirs):
    name = subdir.name
    if xtype == 'tar'  : command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), 'tar -cf {:} -C {:} {:}'.format(destination/'train'/'{:}.tar'.format(name), source / 'train', name))
    elif xtype == 'zip': command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), '(cd {:}; zip -r {:} {:})'.format(source / 'train', destination/'train'/'{:}.zip'.format(name), name))
    else: raise ValueError('invalid compress type : {:}'.format(xtype))


if __name__ == '__main__':
  assert len(sys.argv) == 4, 'invalid argv : {:}'.format(sys.argv)
  source, destination = Path(sys.argv[1]), Path(sys.argv[2])
  main(source, destination, sys.argv[3])
