# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np


# OWN METRICS

def log_likelihood(log_likelihood):
  log_likelihood = np.mean(log_likelihood)
  return log_likelihood

def log_likelihoods(log_likelihoods):
    return np.mean(np.mean(log_likelihoods, axis=1), axis=1)

def outlier_batch(outliers_batch, num_batches):
  outerlier_ratio = outliers_batch / num_batches
  return outerlier_ratio


def outlier_samples(outliers_samples, num_samples):
  outerlier_ratio = outliers_samples / num_samples
  return outerlier_ratio


def mean_responsibilities(responsibilities):
  resp   = np.max(responsibilities, axis=3)
  mean_  = np.mean(resp)
  return mean_


def below_responsibilities(responsibilities, threshold=1.0):
  resp      = np.max(responsibilities, axis=3)
  num_below = np.sum(resp < threshold)
  return num_below / len(resp)


def loss(loss):
  loss = np.mean(loss)
  return loss

# OTHER


def dunn_index(points, labels):
  ''' name wrapper for dunn_fast '''
  return dunn_fast(points, labels)


""" AUTHOR: "Joaquim Viegas"
JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices

dunn(k_list):
    Slow implementation of Dunn index that depends on numpy
    -- basec.pyx Cython implementation is much faster but flower than dunn_fast()
dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
davisbouldin(k_list, k_centers):
    Implementation of Davis Boulding index that depends on numpy
    -- basec.pyx Cython implementation is much faster
"""

from sklearn.metrics.pairwise import euclidean_distances

def delta(ck, cl):
  values = np.ones([len(ck), len(cl)]) * 10000

  for i in range(len(ck)):
    for j in range(len(cl)):
      values[i, j] = np.linalg.norm(ck[i] - cl[j])

  return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(k_list):
    """ Dunn index [CVI]

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas     = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range    = list(range(len(k_list)))

    for k in l_range:
      for l in (l_range[0:k] + l_range[k + 1:]):
          deltas[k, l] = delta(k_list[k], k_list[l])
      big_deltas[k] = big_delta(k_list[k])

    di         = np.min(deltas) / np.max(big_deltas)
    return di


def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]
    return np.min(values)


def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    # values = values[np.nonzero(values)]

    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)

    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances  = euclidean_distances(points)
    ks         = np.sort(np.unique(labels))

    deltas     = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range    = list(range(0, len(ks)))

    for k in l_range:
      for l in (l_range[0:k] + l_range[k + 1:]):
        deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
      big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di         = np.min(deltas) / np.max(big_deltas)
    return di

