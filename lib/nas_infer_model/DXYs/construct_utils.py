import torch
import torch.nn.functional as F


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = x.new_zeros(x.size(0), 1, 1, 1)
    mask = mask.bernoulli_(keep_prob)
    x = torch.div(x, keep_prob)
    x.mul_(mask)
  return x


def return_alphas_str(basemodel):
  if hasattr(basemodel, 'alphas_normal'):
    string = 'normal [{:}] : \n-->>{:}'.format(basemodel.alphas_normal.size(), F.softmax(basemodel.alphas_normal, dim=-1) )
  else: string = ''
  if hasattr(basemodel, 'alphas_reduce'):
    string = string + '\nreduce : {:}'.format( F.softmax(basemodel.alphas_reduce, dim=-1) )

  if hasattr(basemodel, 'get_adjacency'):
    adjacency = basemodel.get_adjacency()
    for i in range( len(adjacency) ):
      weight = F.softmax( basemodel.connect_normal[str(i)], dim=-1 )
      adj = torch.mm(weight, adjacency[i]).view(-1)
      adj = ['{:3.3f}'.format(x) for x in adj.cpu().tolist()]
      string = string + '\nnormal--{:}-->{:}'.format(i, ', '.join(adj))
    for i in range( len(adjacency) ):
      weight = F.softmax( basemodel.connect_reduce[str(i)], dim=-1 )
      adj = torch.mm(weight, adjacency[i]).view(-1)
      adj = ['{:3.3f}'.format(x) for x in adj.cpu().tolist()]
      string = string + '\nreduce--{:}-->{:}'.format(i, ', '.join(adj))

  if hasattr(basemodel, 'alphas_connect'):
    weight = F.softmax(basemodel.alphas_connect, dim=-1).cpu()
    ZERO = ['{:.3f}'.format(x) for x in weight[:,0].tolist()]
    IDEN = ['{:.3f}'.format(x) for x in weight[:,1].tolist()]
    string = string + '\nconnect [{:}] : \n ->{:}\n ->{:}'.format( list(basemodel.alphas_connect.size()), ZERO, IDEN )
  else:
    string = string + '\nconnect = None'
  
  if hasattr(basemodel, 'get_gcn_out'):
    outputs = basemodel.get_gcn_out(True)
    for i, output in enumerate(outputs):
      string = string + '\nnormal:[{:}] : {:}'.format(i, F.softmax(output, dim=-1) )

  return string


def remove_duplicate_archs(all_archs):
  archs = []
  str_archs = ['{:}'.format(x) for x in all_archs]
  for i, arch_x in enumerate(str_archs):
    choose = True
    for j in range(i):
      if arch_x == str_archs[j]:
        choose = False; break
    if choose: archs.append(all_archs[i])
  return archs
