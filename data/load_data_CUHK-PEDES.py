import json

def main():
  xpath = 'caption_all.json'
  with open(xpath, 'r') as cfile:
    cap_data = json.load(cfile)
  print ('There are {:} images'.format( len(cap_data) ))
  IDs = set()
  for idx, data in enumerate( cap_data ):
    IDs.add( data['id'] )
    assert len( data['captions'] ) > 0, 'invalid {:}-th caption length : {:} {:}'.format(idx, data['captions'], len(data['captions']))
  print ('IDs :: min={:}, max={:}, num={:}'.format(min(IDs), max(IDs), len(IDs)))

if __name__ == '__main__':
  main()
