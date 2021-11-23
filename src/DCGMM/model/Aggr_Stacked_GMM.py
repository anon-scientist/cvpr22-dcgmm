#
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
import tensorflow as tf ;
from collections  import defaultdict
from importlib    import import_module
from DCGMM.utils  import log
from .            import Model



class Aggr_Stacked_GMM(Model):

  def __init__(self, **kwargs):
    super(Aggr_Stacked_GMM, self).__init__(**kwargs)
    self.kwargs = kwargs
    self.lookup = dict()
    self.num_tasks = None; self.task_list = None

    self.percentage_task_done = 1.0


  def compile(self):
    """ used for external models, ignored for now """
    pass


  def build(self):
    """ prepares the aggregated model by building the required structures"""
    flags = self.kwargs
    self.task_list = [k for (k, _) in flags.items() if k.startswith('T') and (k[1:]).isnumeric()]
    self.num_tasks = len(self.task_list)

    self._init_structures(flags)


  def _init_structures(self, flags):
    """
      init data structure (gmm reference table)
      StackedGMM mapping: TX -> object, self.lookup holds reference to model under its task key
    """
    gmm_module = import_module(f'model.Stacked_GMM')
    gmm_class = getattr(gmm_module, 'Stacked_GMM')

    if self.task_list and self.num_tasks:
      for k, i in zip(self.task_list, range(0, self.num_tasks)):
        self.lookup[k] = gmm_class(**flags)
        self.lookup[k].build(**flags)
        log.debug(f'INIT Stacked_GMM with key: {k}, object ref.: {self.lookup[k]}')

    self.eval = dict()

  # TODO AK: check if loading of vars for each StackedGMM inside aggregated object is working at all.
  def get_model_variables_store_load(self, **kwargs):
    """ collect all model variables to load or create a checkpoint """
    model_vars = dict()
    for key, gmm in self.lookup.items():
      model_vars.update({key+"/"+var_key : value for var_key, value in gmm.get_model_variables_store_load().items()}) ;
      ##for layer in gmm.layers: # collect all variables from all models layers
      #  model_vars.update({ key : layer.get_layer_variables(all_=True)}) # {[T1 : {L2_gmm : { pis, ... }}, T2 : {...}}

    print (model_vars.keys())
    return model_vars


  def set_parameters(self, **kwargs):
    self.percentage_task_done = kwargs.get('percentage_task_done', 1.0)

  def sample_one_batch(self, topdown=None, last_layer_index=-1):
    """ sample one batch for each task and glue together """
    sampled_tensors =  {}
    for task,gmm in self.lookup.items():
      log.debug(f'Sampling for T{task}, GMM object: ...')
      sampled_tensors[task] = gmm.sample_one_batch(topdown, last_layer_index)

    bigTensor = tf.concat([val for key,val in sampled_tensors.items()], axis=0) ;
    print (bigTensor.shape, "SAAAAAMPLE")

    return bigTensor ;


  def samples_2_numpy(self, sampled, prefix):
    ''' simply a pass-through to the stackedGMM function '''
    for index,(key,gmm) in enumerate(self.lookup.items()):
      gmm.samples_2_numpy(sampled, prefix) ;
      break 

  def vars_2_numpy(self):
    for key, gmm in self.lookup.items():
      for layer in gmm.layers:
        v = layer.get_layer_variables()
        for k, v in v.items():
          np.save('./output_vars/' + (key + '_' + k + '.npy'), v.numpy())


  def train_step(self, xs, ys=None, **kwargs):
    """ train the GMM corresponding to the task number """
    if kwargs['task']:
      task = kwargs['task']
      self.lookup[task].train_step(xs=xs, ys=ys)


  def test(self, test_iterator, **kwargs):
    """ @return: dict(triple(model <str>, source <str>, metric <str>: tuple(metric_value_raw <float>, formatted metric value <str>))) """
    results = defaultdict(list) # saves all per sample test results of all gmm sub-moddels
    batch_acc = list()  # holding batch accuracies calculated from the per-sample log-likelihoods
    return_results = dict() # formatted returning results


    if kwargs.get('task_name',None) == 'DAll': # test ensemble of StackedGMMs
      for xs, ys in test_iterator: # process batch
        gmms_test_results = self.test_step(xs, ys) # returns {gmmKey: test_result_dict}
        stacked_hoods =[] ;
        for gmm_key, test_results in gmms_test_results.items(): #  K models for M class labels
          hoods = test_results['log_likelihoods'] # e.g. { 'T1', 'L2_gmm', 'log_likelihoods' : array[N] } array: N,pY,pX
          results[(gmm_key, 'log_likelihoods')] = hoods ;
          mean_hood = np.mean(hoods)
          results[(gmm_key, 'log_likelihood')] =  mean_hood
          stacked_hoods.append(hoods) ;
        stack = tf.stack(stacked_hoods)

        # Each GMM votes for each sample, winner takes the sample -> performed batch-wise
        # CLASSIFIER here
        ys_true = np.squeeze(np.argmax(ys, axis=1))
        ys_pred = np.squeeze(np.argmax(stack, axis=0))
        xs_acc = np.equal(ys_true, ys_pred)
        xs_acc = np.mean(xs_acc)
        batch_acc.append(xs_acc)

      results[('Aggr_Stacked_GMM', 'gmm_accuracy')] = np.array(batch_acc) ;

    else: # test each sub-model on specific tasks
      for xs, ys in test_iterator:
        gmms_test_results = self.test_step(xs, ys)
        for gmm_key, metric_and_values in gmms_test_results.items(): # {'TX' : {'LX_LAYER' : {'METRIC_NAME': 'METRIC_VALUE'}}, TX+1 ....
            for metric_name, metric_value in metric_and_values.items(): # {'METRIC_NAME': 'METRIC_VALUE'}
              results[(gmm_key, metric_name)] += [np.mean(metric_value)] # { 'METRIC_VALUE' }

    for (gmm_key_layer_name, metric_name), metric_values in results.items(): # format metrics for printing
      if 'accuracy' in metric_name:
        format_str = '{:10.1%}'
      elif 'gmm_accuracy' in metric_name:
        format_str = '{:10.4f}'
      else:
        format_str = '{:10.2f}'
      metric_values = np.mean(metric_values)
      return_results[(gmm_key_layer_name, metric_name)] = (metric_values, format_str.format(metric_values))

    #self.vars_2_numpy()
    return return_results


  def evaluate(self, xs, ys, **kwargs):
    for key, gmm in self.lookup.items():
      gmm.evaluate(xs,ys,**kwargs)


  def test_step(self, xs, ys=None, **kwargs):
    ''' assume that sampling layer is the one to read off loglik from '''
    """ idea here is to call eval on each GMM seperately and collect all return_results in a aggregated structure """

    self.evaluate(xs,ys, **kwargs) ;
    sub_model_results = dict()
    for key, gmm in self.lookup.items():
      sampling_layer_idx = gmm.sampling_layer
      sampling_layer_perf = gmm.eval[sampling_layer_idx] ;
      sub_model_results[key] = sampling_layer_perf ;

    return sub_model_results


  def reset(self, **kwargs):
    """ Each model is trained on a separate task, so reset is not needed """
    pass
