# python ./data/decompress.py $TORCH_HOME/ILSVRC2012-TAR/ ./data/data/ILSVRC2012 tar
# python ./data/decompress.py $TORCH_HOME/ILSVRC2012-ZIP/ ./data/data/ILSVRC2012 zip
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
  #xcmd = '(echo {:} $(date +\"%Y-%h-%d--%T\") \"PID:\"$$; {:}; sleep 0.1s)'.format(prefix, cmd)
  #xcmd = '(echo {:} $(date +\"%Y-%h-%d--%T\") \"PID:\"$$; {:}; sleep 0.1s; pmap $$; echo \"\")'.format(prefix, cmd)
  #xcmd = '(echo {:} $(date +\"%Y-%h-%d--%T\") \"PID:\"$$; {:}; sleep 0.1s; pmap $$; echo \"\")'.format(prefix, cmd)
  xcmd = '(echo {:} $(date +\"%Y-%h-%d--%T\") \"PID:\"$$; {:}; sleep 0.1s)'.format(prefix, cmd)
  return xcmd


def mkILSVRC2012(destination):
  destination = destination.resolve()
  destination.mkdir(parents=True, exist_ok=True)
  os.system('rm -rf {:}'.format(destination))
  destination.mkdir(parents=True, exist_ok=True)
  (destination/'train').mkdir(parents=True, exist_ok=True)


def main(source, destination, xtype):
  assert source.exists(), '{:} does not exist'.format(source)
  assert (source/'train'  ).exists(), '{:}/train does not exist'.format(source)
  if xtype == 'tar'  : assert (source/'val.tar').exists(), '{:}/val   does not exist'.format(source)
  elif xtype == 'zip': assert (source/'val.zip').exists(), '{:}/val   does not exist'.format(source)
  else               : raise ValueError('invalid unzip type : {:}'.format(xtype))
  #assert num_process > 0, 'invalid num_process : {:}'.format(num_process)
  source      = source.resolve()
  mkILSVRC2012(destination)

  subdirs = list( (source / 'train').glob('n*') )
  all_commands = []
  assert len(subdirs) == 1000, 'ILSVRC2012 should contain 1000 classes instead of {:}.'.format( len(subdirs) )
  for idx, subdir in enumerate(subdirs):
    name = subdir.name
    if xtype == 'tar'  : cmd = command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), 'tar -xf {:} -C {:}'.format(source/'train'/'{:}'.format(name), destination / 'train'))
    elif xtype == 'zip': cmd = command('{:03d}/{:03d}-th: '.format(idx, len(subdirs)), 'unzip -qd {:} {:}'.format(destination / 'train', source/'train'/'{:}'.format(name)))
    else               : raise ValueError('invalid unzip type : {:}'.format(xtype))
    all_commands.append( cmd )
  if xtype == 'tar'  : cmd = command('', 'tar -xf {:} -C {:}'.format(source/'val.tar', destination))
  elif xtype == 'zip': cmd = command('', 'unzip -qd {:} {:}'.format(destination, source/'val.zip'))
  else               : raise ValueError('invalid unzip type : {:}'.format(xtype))
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
  #num_process = int(sys.argv[3])
  if sys.argv[3] == 'wget':
    with open(source) as f:
      content = f.readlines()
    content = [x.strip() for x in content]
    assert len(content) == 1000, 'invalid lines={:} from {:}'.format( len(content), source )
    mkILSVRC2012(destination)
    all_commands = []
    cmd = command('make-val', 'wget -q http://10.127.2.44:8000/ILSVRC2012-TAR/val.tar --directory-prefix={:} ; tar -xf {:} -C {:} ; rm {:}'.format(destination, destination / 'val.tar', destination, destination / 'val.tar'))
    all_commands.append(cmd)
    for idx, name in enumerate(content):
      cmd = command('{:03d}/{:03d}-th: '.format(idx, len(content)), 'wget -q http://10.127.2.44:8000/ILSVRC2012-TAR/train/{:}.tar --directory-prefix={:} ; tar -xf {:}.tar -C {:} ; rm {:}.tar'.format(name, destination / 'train', destination / 'train' / name, destination / 'train', destination / 'train' / name))
      all_commands.append(cmd)
    for i, cmd in enumerate(all_commands): print(cmd)
  else:
    main(source, destination, sys.argv[3])
