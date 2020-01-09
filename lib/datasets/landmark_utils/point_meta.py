# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy, math, torch, numpy as np
from xvision import normalize_points
from xvision import denormalize_points


class PointMeta():
  # points    : 3 x num_pts (x, y, oculusion)
  # image_size: original [width, height]
  def __init__(self, num_point, points, box, image_path, dataset_name):

    self.num_point = num_point
    if box is not None:
      assert (isinstance(box, tuple) or isinstance(box, list)) and len(box) == 4
      self.box = torch.Tensor(box)
    else: self.box = None
    if points is None:
      self.points = points
    else:
      assert len(points.shape) == 2 and points.shape[0] == 3 and points.shape[1] == self.num_point, 'The shape of point is not right : {}'.format( points )
      self.points = torch.Tensor(points.copy())
    self.image_path = image_path
    self.datasets = dataset_name

  def __repr__(self):
    if self.box is None: boxstr = 'None'
    else               : boxstr = 'box=[{:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(*self.box.tolist())
    return ('{name}(points={num_point}, '.format(name=self.__class__.__name__, **self.__dict__) + boxstr + ')')

  def get_box(self, return_diagonal=False):
    if self.box is None: return None
    if not return_diagonal:
      return self.box.clone()
    else:
      W = (self.box[2]-self.box[0]).item()
      H = (self.box[3]-self.box[1]).item()
      return math.sqrt(H*H+W*W)

  def get_points(self, ignore_indicator=False):
    if ignore_indicator: last = 2
    else               : last = 3
    if self.points is not None: return self.points.clone()[:last, :]
    else                      : return torch.zeros((last, self.num_point))

  def is_none(self):
    #assert self.box is not None, 'The box should not be None'
    return self.points is None
    #if self.box is None: return True
    #else               : return self.points is None

  def copy(self):
    return copy.deepcopy(self)

  def visiable_pts_num(self):
    with torch.no_grad():
      ans = self.points[2,:] > 0
      ans = torch.sum(ans)
      ans = ans.item()
    return ans
  
  def special_fun(self, indicator):
    if indicator == '68to49': # For 300W or 300VW, convert the default 68 points to 49 points.
      assert self.num_point == 68, 'num-point must be 68 vs. {:}'.format(self.num_point)
      self.num_point = 49
      out = torch.ones((68), dtype=torch.uint8)
      out[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,60,64]] = 0
      if self.points is not None: self.points = self.points.clone()[:, out]
    else:
      raise ValueError('Invalid indicator : {:}'.format( indicator ))

  def apply_horizontal_flip(self):
    #self.points[0, :] = width - self.points[0, :] - 1
    # Mugsy spefic or Synthetic
    if self.datasets.startswith('HandsyROT'):
      ori = np.array(list(range(0, 42)))
      pos = np.array(list(range(21,42)) + list(range(0,21)))
      self.points[:, pos] = self.points[:, ori]
    elif self.datasets.startswith('face68'):
      ori = np.array(list(range(0, 68)))
      pos = np.array([17,16,15,14,13,12,11,10, 9, 8,7,6,5,4,3,2,1, 27,26,25,24,23,22,21,20,19,18, 28,29,30,31, 36,35,34,33,32, 46,45,44,43,48,47, 40,39,38,37,42,41, 55,54,53,52,51,50,49,60,59,58,57,56,65,64,63,62,61,68,67,66])-1
      self.points[:, ori] = self.points[:, pos]
    else:
      raise ValueError('Does not support {:}'.format(self.datasets))



# shape = (H,W)
def apply_affine2point(points, theta, shape):
  assert points.size(0) == 3, 'invalid points shape : {:}'.format(points.size())
  with torch.no_grad():
    ok_points = points[2,:] == 1
    assert torch.sum(ok_points).item() > 0, 'there is no visiable point'
    points[:2,:] = normalize_points(shape, points[:2,:])

    norm_trans_points = ok_points.unsqueeze(0).repeat(3, 1).float()

    trans_points, ___ = torch.gesv(points[:, ok_points], theta)

    norm_trans_points[:, ok_points] = trans_points
    
  return norm_trans_points



def apply_boundary(norm_trans_points):
  with torch.no_grad():
    norm_trans_points = norm_trans_points.clone()
    oks = torch.stack((norm_trans_points[0]>-1, norm_trans_points[0]<1, norm_trans_points[1]>-1, norm_trans_points[1]<1, norm_trans_points[2]>0))
    oks = torch.sum(oks, dim=0) == 5
    norm_trans_points[2, :] = oks
  return norm_trans_points
