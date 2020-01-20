import os, sys, math, time, random, argparse
import tensorflow as tf
from pathlib import Path


def test_a():
  x = tf.Variable([[1.], [2.], [4.0]])
  with tf.GradientTape(persistent=True) as g:
    trn = tf.math.exp(tf.math.reduce_sum(x))
    val = tf.math.cos(tf.math.reduce_sum(x))
    dT_dx = g.gradient(trn, x)
    dV_dx = g.gradient(val, x)
    hess_vector = g.gradient(dT_dx, x, output_gradients=dV_dx)
  print ('calculate ok : {:}'.format(hess_vector))

def test_b():
  cce = tf.keras.losses.SparseCategoricalCrossentropy()
  L1 = tf.convert_to_tensor([0, 1, 2])
  L2 = tf.convert_to_tensor([2, 0, 1])
  B = tf.Variable([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
  with tf.GradientTape(persistent=True) as g:
    trn = cce(L1, B)
    val = cce(L2, B)
    dT_dx = g.gradient(trn, B)
    dV_dx = g.gradient(val, B)
    hess_vector = g.gradient(dT_dx, B, output_gradients=dV_dx)
  print ('calculate ok : {:}'.format(hess_vector))

def test_c():
  cce = tf.keras.losses.CategoricalCrossentropy()
  L1 = tf.convert_to_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
  L2 = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
  B = tf.Variable([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
  with tf.GradientTape(persistent=True) as g:
    trn = cce(L1, B)
    val = cce(L2, B)
    dT_dx = g.gradient(trn, B)
    dV_dx = g.gradient(val, B)
    hess_vector = g.gradient(dT_dx, B, output_gradients=dV_dx)
  print ('calculate ok : {:}'.format(hess_vector))

if __name__ == '__main__':
  print(tf.__version__)
  test_c()
  #test_b()
  #test_a()
