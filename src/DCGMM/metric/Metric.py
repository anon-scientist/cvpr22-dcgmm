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
import sklearn.metrics
import inspect
from DCGMM.utils            import log
from importlib              import import_module
from enum                   import Flag

class ArgEnum(Flag): # TODO: remove the ArgEnum
  def __str__(self):
    return self.name.lower()

  def __repr__(self):
    return str(self)

  @staticmethod
  def arg(s):
    try            : return ArgEnum[s.upper()]
    except KeyError: return s


class Metrics(ArgEnum):
  ACCURACY_SCORE                     = 'accuracy_score',                    # metrics.accuracy_score(y_true, y_pred)           TESTED # Accuracy classification score. e.g., y_true=[1,2,3], y_pred=[1,1,3]
  AUC                                = 'auc',                               # metrics.auc(x, y)                                       # Compute Area Under the Curve (AUC) using the trapezoidal rule
  AVERAGE_PRECISION_SCORE            = 'average_precision_score',           # metrics.average_precision_score(y_true, y_score)        # Compute average precision (AP) from prediction scores
  BALANCED_ACCURACY_SCORE            = 'balanced_accuracy_score',           # metrics.balanced_accuracy_score(y_true, y_pred)  TESTED # Compute the balanced accuracy
  BRIER_SCORE_LOSS                   = 'brier_score_loss',                  # metrics.brier_score_loss(y_true, y_prob)                # Compute the Brier score.
  CLASSIFICATION_REPORT              = 'classification_report',             # metrics.classification_report(y_true, y_pred)    TESTED # Build a text report showing the main classification metrics; raise Exception("remove this semicolon")
  COHEN_KAPPE_SCORE                  = 'cohen_kappa_score',                 # metrics.cohen_kappa_score(y1, y2)                       # Cohens kappa: a statistic that measures inter-annotator agreement.
  CONFUSION_MATRIX                   = 'confusion_matrix',                  # metrics.confusion_matrix(y_true, y_pred)                # Compute confusion matrix to evaluate the accuracy of a classification
  F1_SCORE                           = 'f1_score',                          # metrics.f1_score(y_true, y_pred)                 TESTED # Compute the F1 score, also known as balanced F-score or F-measure
  FBETA_SCORE                        = 'fbeta_score',                       # metrics.fbeta_score(y_true, y_pred, beta)               # Compute the F-beta score
  HAMMING_LOSS                       = 'hamming_loss',                      # metrics.hamming_loss(y_true, y_pred)             TESTED # Compute the average Hamming loss.
  HINGE_LOSS                         = 'hinge_loss',                        # metrics.hinge_loss(y_true, pred_decision)               # Average hinge loss (non-regularized)
  JACCARD_SCORE                      = 'jaccard_score',                     # metrics.jaccard_score(y_true, y_pred)            TESTED # Jaccard similarity coefficient score
  LOG_LOSS                           = 'log_loss',                          # metrics.log_loss(y_true, y_pred)                        # Log loss, aka logistic loss or cross-entropy loss.
  MATTHEWS_CORRCOEF                  = 'matthews_corrcoef',                 # metrics.matthews_corrcoef(y_true, y_pred)        TESTED # Compute the Matthews correlation coefficient (MCC)
  PRECISION_RECALL_CURVE             = 'precision_recall_curve',            # metrics.precision_recall_curve(y_true)                  # Compute precision-recall pairs for different probability thresholds
  PRECISION_RECALL_FSCORE_SUPPORT    = 'precision_recall_fscore_support',   # metrics.precision_recall_fscore_support()        TESTED # Compute precision, recall, F-measure and support for each class
  PRECISION_SCORE                    = 'precision_score',                   # metrics.precision_score(y_true, y_pred)          TESTED # Compute the precision
  RECALL_SCORE                       = 'recall_score',                      # metrics.recall_score(y_true, y_pred)             TESTED # Compute the recall
  ROC_AUC_SCORE                      = 'roc_auc_score',                     # metrics.roc_auc_score(y_true, y_score)                  # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
  ROC_CURVE                          = 'roc_curve',                         # metrics.roc_curve(y_true, y_score)               TESTED # Compute Receiver operating characteristic (ROC)
  ZERO_ONE_LOSS                      = 'zero_one_loss',                     # metrics.zero_one_loss(y_true, y_pred)            TESTED # Zero-one classification loss.
  # Regression metrics
  EXPLAINED_VARIANCE_SCORE           = 'explained_variance_score',          # metrics.explained_variance_score(y_true, y_pred)        # Explained variance regression score function
  MAX_ERROR                          = 'max_error',                         # metrics.max_error(y_true, y_pred)                       # max_error metric calculates the maximum residual error.
  MEAN_ABSOLUTE_ERROR                = 'mean_absolute_error',               # metrics.mean_absolute_error(y_true, y_pred)             # Mean absolute error regression loss
  MEAN_SQUARED_ERROR                 = 'mean_squared_error',                # metrics.mean_squared_error(y_true, y_pred)              # Mean squared error regression loss
  MEAN_SQUARED_LOG_ERROR             = 'mean_squared_log_error',            # metrics.mean_squared_log_error(y_true, y_pred)          # Mean squared logarithmic error regression loss
  MEDIAN_ABSOLUTE_ERROR              = 'median_absolute_error',             # metrics.median_absolute_error(y_true, y_pred)           # Median absolute error regression loss
  R2_SCORE                           = 'r2_score',                          # metrics.r2_score(y_true, y_pred)                        # R^2 (coefficient of determination) regression score function.
  # Clustering metrics (un- and supervised)
  ADJUSTED_MUTUAL_INFO_SCORE         = 'adjusted_mutual_info_score'         # metrics.adjusted_mutual_info_score()                    # adjusted mutual information between two clusterings.
  ADJUSTED_RAND_SCORE                = 'adjusted_rand_score'                # metrics.adjusted_rand_score(labels_true)                # rand index adjusted for chance.
  CALINSKI_HARABASZ_SCORE            = 'calinski_harabasz_score'            # metrics.calinski_harabasz_score(x, labels)              # compute the calinski and harabasz score.
  DAVIES_BOULDIN_SCORE               = 'davies_bouldin_score'               # metrics.davies_bouldin_score(x, labels)        TESTED   # computes the davies-bouldin score.
  COMPLETENESS_SCORE                 = 'completeness_score'                 # metrics.completeness_score(labels_true)                 # completeness metric of a cluster labeling given a ground truth.
  FOWLKES_MALLOWS_SCORE              = 'fowlkes_mallows_score'              # metrics.fowlkes_mallows_score(labels_true)              # measure the similarity of two clusterings of a set of points.
  HOMOGENEITY_COMPLETENESS_V_MEASURE = 'homogeneity_completeness_v_measure' # metrics.homogeneity_completeness_v_measure()            # compute the homogeneity and completeness and v-measure scores at once.
  HOMOGENEITY_SCORE                  = 'homogeneity_score'                  # metrics.homogeneity_score(labels_true)                  # homogeneity metric of a cluster labeling given a ground truth.
  MUTUAL_INFO_SCORE                  = 'mutual_info_score'                  # metrics.mutual_info_score(labels_true)                  # mutual information between two clusterings.
  NORMALIZED_MUTUAL_INFO_SCORE       = 'normalized_mutual_info_score'       # metrics.normalized_mutual_info_score()                  # normalized mutual information between two clusterings.
  SILHOUETTE_SCORE                   = 'silhouette_score'                   # metrics.silhouette_score(x, labels)                     # compute the mean silhouette coefficient of all samples.
  SILHOUETTE_SAMPLES                 = 'silhouette_samples'                 # metrics.silhouette_samples(X, LABELS)                   # compute the silhouette coefficient for each sample.
  V_MEASURE_SCORE                    = 'v_measure_score'                    # metrics.v_measure_score(labels_true)                    # V-measure cluster labeling given a ground truth.

  # not from sklearn.metrics
  DUNN_INDEX                         = 'dunn_index'                         # https://github.com/jqmviegas/jqm_cvi                    # Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
  LOG_LIKELIHOOD                     = 'log_likelihood'
  LOG_LIKELIHOODS                    = 'log_likelihoods'
  OUTLIER_BATCH                      = 'outlier_batch'
  OUTLIER_SAMPLES                    = 'outlier_samples'
  MEAN_RESPONSIBILITIES              = 'mean_responsibilities'
  BELOW_RESPONSIBILITIES             = 'below_responsibilities'


metrics = list(Metrics)

class Metric(object):

  def __init__(self, parameter):
    ''' metric class provide metric functions from scikit-learn (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
      New: own modules
    '''
    if not isinstance(parameter.metrics, list):
      parameter.metrics = [parameter.metrics]
    self.metrics    = parameter.metrics

    self.csv_header = self.build_csv_header()
    self.metric_parameter()


  def build_csv_header(self):
    header  = ['task', 'iteration']
    header += [f'{metric}' for metric in self.metrics]
    return header


  def metric_parameter(self):
    s = ''
    for metric in self.metrics:
      s                += f'metric {metric}: '
      try   : metric_function = getattr(sklearn.metrics, str(metric))
      except: metric_function = getattr(import_module(f'DCGMM.metric.other'), f'{metric}') # if not a sklearn metric

      params            = inspect.getfullargspec(metric_function)
      args , varargs    = params[0], params[1]
      varkw, defaults   = params[2], params[3]
      num_args          = len(args)
      num_default_args  = len(defaults) if defaults else 0
      arg_list          = ', '.join([ arg                       for arg          in args[:num_args - num_default_args] ])
      defaults          = defaults if defaults else list()
      default_arg_list  = ', '.join([''] + [ f'{arg}={default}' for arg, default in zip(args[num_args - num_default_args:], defaults) ])
      s                += arg_list
      s                += default_arg_list    if num_default_args != num_args else ''
      s                += f', args={varargs}' if varargs                      else ''
      s                += f', kwargs={varkw}' if varkw                        else ''
      s                += ', '
    log.info(s)
    return s


  def eval(self, **kwargs):
    ''' calculate the specified metric(s)

    @param print_pre : is printed befor the metric output
    @param print_post: is printed after the metric output
    @param kwargs    : parameter for metric calculation (look at the metric function documentation)
      https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics, e.g.:
        y_true (ground truth) (np.array)                                                                        ,
        y_pred (predicted labels) (np.array)                                                                    ,
        special (metric specific parameter (dict )), e.g. special={Metrics.ACCURACY_SCORE: dict(normalize=True)},
        list: if True, a list of measurement values is returned (list(values, ...)) (default if both False)
          AND
        dict: if True, a dict of measurement values is returned (dict(Metrics=value)) (default=False)
    @return: the return values of the metric function(s) ((optional)list, (optional)dict)
    '''
    values      = list()
    values_dict = dict()
    s           = f'{kwargs.get("print_pre", ""):<35}'
    for metric in self.metrics:
      try   : metric_function = getattr(sklearn.metrics, str(metric))
      except: metric_function = getattr(import_module(f'DCGMM.metric.other'), f'{metric}') # if not a sklearn metric
      param_list        = inspect.getfullargspec(metric_function)[0]                                                # get the valid parameter names of the function
      if metric == str(Metrics.ACCURACY_SCORE): param_list = ['y_true', 'y_pred']                                   # TODO: Workaround. inspect did not return parameter names for accuracy_score function (sklearn==0.23.2 and python==3.6.9)
      format_str            = kwargs.get('special', {}).get(metric, {}).pop('format_str', None)                           # fetch a format string
      filled_param_list     = { param_name: kwargs.get(param_name) for param_name in param_list if param_name in kwargs } # build dict with parameter name and values
      filled_param_list.update(kwargs.get('special', {}).get(metric, {}))                                             # update with special parameter
      metric_value  = metric_function(**filled_param_list)                                                        # call metric function
      metric_value_formated = format_str.format(metric_value) if format_str else metric_value
      values               += [metric_value]                                                                              # combine all metric values
      if format_str: values_dict.update({metric: (metric_value, metric_value_formated)})
      else         : values_dict.update({metric: metric_value})
      # TODO: format multiline metrics for csv files3
      if str(metric) in ['classification_report', 'confusion_matrix']:  # for metrics with multiple output lines
        s += f'\n{metric}\n{metric_value}\n'
      else:
        s += f'| {metric}={metric_value} '
    s      += f'{kwargs.get("print_post", ""):<35}'
    #print(s)
    list_, dict_ = [ kwargs.get(key, False) for key in ['list', 'dict'] ]
    if list_ and dict_: return values, values_dict
    if dict_          : return values_dict
    return values
