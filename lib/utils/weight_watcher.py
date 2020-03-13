#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.03 #
#####################################################
# Reformulate the codes in https://github.com/CalculatedContent/WeightWatcher
#####################################################
import numpy as np
from typing import List
import torch.nn as nn
from collections import OrderedDict
from sklearn.decomposition import TruncatedSVD


def available_module_types():
  return (nn.Conv2d, nn.Linear)


def get_conv2D_Wmats(tensor: np.ndarray) -> List[np.ndarray]:
  """
  Extract W slices from a 4 index conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).
  Return ij (N x M) matrices
  """
  mats = []
  N, M, imax, jmax = tensor.shape
  assert N + M >= imax + jmax, 'invalid tensor shape detected: {}x{} (NxM), {}x{} (i,j)'.format(N, M, imax, jmax)
  for i in range(imax):
    for j in range(jmax):
      w = tensor[:, :, i, j]
      if N < M: w = w.T
      mats.append(w)
  return mats


def glorot_norm_check(W, N, M, rf_size, lower=0.5, upper=1.5):
  """Check if this layer needs Glorot Normalization Fix"""

  kappa = np.sqrt(2 / ((N + M) * rf_size))
  norm = np.linalg.norm(W)

  check1 = norm / np.sqrt(N * M)
  check2 = norm / (kappa * np.sqrt(N * M))

  if (rf_size > 1) and (check2 > lower) and (check2 < upper):
    return check2, True
  elif (check1 > lower) & (check1 < upper):
    return check1, True
  else:
    if rf_size > 1: return check2, False
    else: return check1, False

def glorot_norm_fix(w, n, m, rf_size):
  """Apply Glorot Normalization Fix."""
  kappa = np.sqrt(2 / ((n + m) * rf_size))
  w = w / kappa
  return w


def analyze_weights(weights, min_size, max_size, alphas, lognorms, spectralnorms, softranks, normalize, glorot_fix):
  results = OrderedDict()
  count = len(weights)
  if count == 0: return results

  for i, weight in enumerate(weights):
    M, N = np.min(weight.shape), np.max(weight.shape)
    Q = N / M
    results[i] = cur_res = OrderedDict(N=N, M=M, Q=Q)
    check, checkTF = glorot_norm_check(weight, N, M, count)
    cur_res['check'] = check
    cur_res['checkTF'] = checkTF
    # assume receptive field size is count
    if glorot_fix:
      weight = glorot_norm_fix(weight, N, M, count)
    else:
      # probably never needed since we always fix for glorot
      weight = weight * np.sqrt(count / 2.0)

    if spectralnorms:  # spectralnorm is the max eigenvalues
      svd = TruncatedSVD(n_components=1, n_iter=7, random_state=10)
      svd.fit(weight)
      sv = svd.singular_values_
      sv_max = np.max(sv)
      if normalize:
        evals = sv * sv / N
      else:
        evals = sv * sv
      lambda0 = evals[0]
      cur_res["spectralnorm"] = lambda0
      cur_res["logspectralnorm"] = np.log10(lambda0)
    else:
      lambda0 = None

    if M < min_size:
      summary = "Weight matrix {}/{} ({},{}): Skipping: too small (<{})".format(i + 1, count, M, N, min_size)
      cur_res["summary"] = summary
      continue
    elif max_size > 0 and M > max_size:
      summary = "Weight matrix {}/{} ({},{}): Skipping: too big (testing) (>{})".format(i + 1, count, M, N, max_size)
      cur_res["summary"] = summary
      continue
    else:
      summary = []
    if alphas:
      import powerlaw
      svd = TruncatedSVD(n_components=M - 1, n_iter=7, random_state=10)
      svd.fit(weight.astype(float))
      sv = svd.singular_values_
      if normalize: evals = sv * sv / N
      else: evals = sv * sv

      lambda_max = np.max(evals)
      fit = powerlaw.Fit(evals, xmax=lambda_max, verbose=False)
      alpha = fit.alpha
      cur_res["alpha"] = alpha
      D = fit.D
      cur_res["D"] = D
      cur_res["lambda_min"] = np.min(evals)
      cur_res["lambda_max"] = lambda_max
      alpha_weighted = alpha * np.log10(lambda_max)
      cur_res["alpha_weighted"] = alpha_weighted
      tolerance = lambda_max * M * np.finfo(np.max(sv)).eps
      cur_res["rank_loss"] = np.count_nonzero(sv > tolerance, axis=-1)

      logpnorm = np.log10(np.sum([ev ** alpha for ev in evals]))
      cur_res["logpnorm"] = logpnorm

      summary.append(
        "Weight matrix {}/{} ({},{}): Alpha: {}, Alpha Weighted: {}, D: {}, pNorm {}".format(i + 1, count, M, N, alpha,
                                                                                             alpha_weighted, D,
                                                                                             logpnorm))

    if lognorms:
      norm = np.linalg.norm(weight)  # Frobenius Norm
      cur_res["norm"] = norm
      lognorm = np.log10(norm)
      cur_res["lognorm"] = lognorm

      X = np.dot(weight.T, weight)
      if normalize: X = X / N
      normX = np.linalg.norm(X)  # Frobenius Norm
      cur_res["normX"] = normX
      lognormX = np.log10(normX)
      cur_res["lognormX"] = lognormX

      summary.append(
        "Weight matrix {}/{} ({},{}): LogNorm: {} ; LogNormX: {}".format(i + 1, count, M, N, lognorm, lognormX))

      if softranks:
        softrank = norm ** 2 / sv_max ** 2
        softranklog = np.log10(softrank)
        softranklogratio = lognorm / np.log10(sv_max)
        cur_res["softrank"] = softrank
        cur_res["softranklog"] = softranklog
        cur_res["softranklogratio"] = softranklogratio
        summary += "{}. Softrank: {}. Softrank log: {}. Softrank log ratio: {}".format(summary, softrank, softranklog,
                                                                                       softranklogratio)
    cur_res["summary"] = "\n".join(summary)
  return results


def compute_details(results):
  """
  Return a pandas data frame.
  """
  final_summary = OrderedDict()

  metrics = {
    # key in "results" : pretty print name
    "check": "Check",
    "checkTF": "CheckTF",
    "norm": "Norm",
    "lognorm": "LogNorm",
    "normX": "Norm X",
    "lognormX": "LogNorm X",
    "alpha": "Alpha",
    "alpha_weighted": "Alpha Weighted",
    "spectralnorm": "Spectral Norm",
    "logspectralnorm": "Log Spectral Norm",
    "softrank": "Softrank",
    "softranklog": "Softrank Log",
    "softranklogratio": "Softrank Log Ratio",
    "sigma_mp": "Marchenko-Pastur (MP) fit sigma",
    "numofSpikes": "Number of spikes per MP fit",
    "ratio_numofSpikes": "aka, percent_mass, Number of spikes / total number of evals",
    "softrank_mp": "Softrank for MP fit",
    "logpnorm": "alpha pNorm"
  }

  metrics_stats = []
  for metric in metrics:
    metrics_stats.append("{}_min".format(metric))
    metrics_stats.append("{}_max".format(metric))
    metrics_stats.append("{}_avg".format(metric))

    metrics_stats.append("{}_compound_min".format(metric))
    metrics_stats.append("{}_compound_max".format(metric))
    metrics_stats.append("{}_compound_avg".format(metric))

  columns = ["layer_id", "layer_type", "N", "M", "layer_count", "slice",
             "slice_count", "level", "comment"] + [*metrics] + metrics_stats

  metrics_values = {}
  metrics_values_compound = {}

  for metric in metrics:
    metrics_values[metric] = []
    metrics_values_compound[metric] = []

  layer_count = 0
  for layer_id, result in results.items():
    layer_count += 1

    layer_type = np.NAN
    if "layer_type" in result:
      layer_type = str(result["layer_type"]).replace("LAYER_TYPE.", "")

    compounds = {}  # temp var
    for metric in metrics:
      compounds[metric] = []

    slice_count, Ntotal, Mtotal = 0, 0, 0
    for slice_id, summary in result.items():
      if not str(slice_id).isdigit():
        continue
      slice_count += 1

      N = np.NAN
      if "N" in summary:
        N = summary["N"]
        Ntotal += N

      M = np.NAN
      if "M" in summary:
        M = summary["M"]
        Mtotal += M

      data = {"layer_id": layer_id, "layer_type": layer_type, "N": N, "M": M, "slice": slice_id, "level": "SLICE",
              "comment": "Slice level"}
      for metric in metrics:
        if metric in summary:
          value = summary[metric]
          if value is not None:
            metrics_values[metric].append(value)
            compounds[metric].append(value)
            data[metric] = value

    data = {"layer_id": layer_id, "layer_type": layer_type, "N": Ntotal, "M": Mtotal, "slice_count": slice_count,
            "level": "LAYER", "comment": "Layer level"}
    # Compute the compound value over the slices
    for metric, value in compounds.items():
      count = len(value)
      if count == 0:
        continue

      compound = np.mean(value)
      metrics_values_compound[metric].append(compound)
      data[metric] = compound

  data = {"layer_count": layer_count, "level": "NETWORK", "comment": "Network Level"}
  for metric, metric_name in metrics.items():
    if metric not in metrics_values or len(metrics_values[metric]) == 0:
      continue

    values = metrics_values[metric]
    minimum = min(values)
    maximum = max(values)
    avg = np.mean(values)
    final_summary[metric] = avg
    # print("{}: min: {}, max: {}, avg: {}".format(metric_name, minimum, maximum, avg))
    data["{}_min".format(metric)] = minimum
    data["{}_max".format(metric)] = maximum
    data["{}_avg".format(metric)] = avg

    values = metrics_values_compound[metric]
    minimum = min(values)
    maximum = max(values)
    avg = np.mean(values)
    final_summary["{}_compound".format(metric)] = avg
    # print("{} compound: min: {}, max: {}, avg: {}".format(metric_name, minimum, maximum, avg))
    data["{}_compound_min".format(metric)] = minimum
    data["{}_compound_max".format(metric)] = maximum
    data["{}_compound_avg".format(metric)] = avg

  return final_summary


def analyze(model: nn.Module, min_size=50, max_size=0,
            alphas: bool = False, lognorms: bool = True, spectralnorms: bool = False,
            softranks: bool = False, normalize: bool = False, glorot_fix: bool = False):
  """
  Analyze the weight matrices of a model.
  :param model: A PyTorch model
  :param min_size: The minimum weight matrix size to analyze.
  :param max_size: The maximum weight matrix size to analyze (0 = no limit).
  :param alphas: Compute the power laws (alpha) of the weight matrices.
    Time consuming so disabled by default (use lognorm if you want speed)
  :param lognorms: Compute the log norms of the weight matrices.
  :param spectralnorms: Compute the spectral norm (max eigenvalue) of the weight matrices.
  :param softranks: Compute the soft norm (i.e. StableRank) of the weight matrices.
  :param normalize: Normalize or not.
  :param glorot_fix:
  :return: (a dict of all layers' results, a dict of the summarized info)
  """
  names, modules = [], []
  for name, module in model.named_modules():
    if isinstance(module, available_module_types()):
      names.append(name)
      modules.append(module)
  # print('There are {:} layers to be analyzed in this model.'.format(len(modules)))
  all_results = OrderedDict()
  for index, module in enumerate(modules):
    if isinstance(module, nn.Linear):
      weights = [module.weight.cpu().detach().numpy()]
    else:
      weights = get_conv2D_Wmats(module.weight.cpu().detach().numpy())
    results = analyze_weights(weights, min_size, max_size, alphas, lognorms, spectralnorms, softranks, normalize, glorot_fix)
    results['id'] = index
    results['type'] = type(module)
    all_results[index] = results
  summary = compute_details(all_results)
  return all_results, summary