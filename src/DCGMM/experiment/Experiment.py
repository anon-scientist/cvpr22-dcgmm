'''
TODO automatic pulling for loss/forwqrds when model is constructed by hand: call loss on last layer, will compute all prev layer outputs automatically
TODO linear_classifier_layer topdown should depend on energy: mse or ce
DONE  log train eval measures in SGMM_Outliers upon parameter
TODO check --verbose parameter is passed on to kwargs everywhere
TODO have a system that helps restrict experiment instances to certain classes of models
TODO eval script for json logs
TODO rigid parameter management for replay_alex: nr of generated samples;
      mixing proportion, etc etc.
TODO system for a layer having more than one input. important for condsampling
     in dcgmms, while still having good class acuracy. Idea: provide constructor
     and forward_multi in Layer, which either calls subclasses forwrd(xs)
     or overwrites forward_multi
TODO introduce merge_layer thqt (duh) merges qctivities from previous layers, 
     and feeds back control signals. would simplyfy linear_classifier_layer     
TODO check whether Aggregated_GMM works everywhere!

'''
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

import sys
import math
import DCGMM.utils.GPU_Test # import check if GPU is available, disable TensorFlow logging, enable memory grow
import numpy            as np
from importlib          import import_module

from DCGMM.utils              import log, change_loglevel
from DCGMM.measuring          import Logging
from DCGMM.connection_handler import Connection_Handler
from DCGMM.dataset            import Dataset_Type, TF2_Dataset as Dataset
from DCGMM.parsers            import Kwarg_Parser
from DCGMM.parsers            import Command_Line_Parser


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment():
  ''' defines a single experiment with different tasks and parameters '''

  def _init_parser(self, **kwargs):
    ''' initialize command line parser '''
    log.info('init parser')

    command_line_params   = Command_Line_Parser().parse_args()
    self.parser           = Kwarg_Parser(external_arguments=command_line_params, verbose=True,**kwargs)

    #--------------------------------------------------------------------LOGGING
    self.log_level        = self.parser.add_argument('--log_level'                , type=str  , default='INFO', choices=['DEBUG', 'INFO'], help='enable printing and saving')
    #------------------------------------------------------------------- DATASET
    self.results_dir      = self.parser.add_argument('--results_dir'              , type=str  , default='./results'                      , help='set the default directory to search for dataset files')
    self.dataset_dir      = self.parser.add_argument('--dataset_dir'              , type=str  , default='./datasets'                     , help='set the default directory to search for dataset files')
    self.dataset_file     = self.parser.add_argument('--dataset_file'             , type=str  , default='MNIST'                          , help='load a compressed pickle file. If not present, a download attempt is made. This may take a while for large datasets such as SVHN')
    self.dataset_name     = self.parser.add_argument('--dataset_name'             , type=str  , default='mnist'                          , help='load a dataset via tfds. If not present, a download attempt is made. ')
    #------------------------------------------------------------------ TRAINING
    self.epochs           = self.parser.add_argument('--epochs'                   , type=float, default=10                               , help='number of training epochs per taks')
    self.batch_size       = self.parser.add_argument('--batch_size'               , type=int  , default=100                              , help='size of mini-batches we feed from train dataSet.')
    #------------------------------------------------------------------- TESTING
    self.test_batch_size  = self.parser.add_argument('--test_batch_size'          , type=int  , default=100                              , help='batch size for testing')
    self.measuring_points = self.parser.add_argument('--measuring_points'         , type=int  , default=10                               , help='measure X+1 points per task')
    #------------------------------------------------------------- VISUALIZATION
    self.vis_points       = self.parser.add_argument('--vis_points'               , type=int  , default=100                              , help='X visualization points. At individual data of the layers are provided via socket')
    #--------------------------------------------------------- SLT CONFIGURATION
    self.DAll             = self.parser.add_argument('--DAll'                     , type=str  , default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   , help='for each task test all given classes')
    #---------------------------------------------------------------- CHECKPOINT
    self.exp_id           = self.parser.add_argument('--exp_id'                   , type=str  , default='0'                              , help='unique experiment id (for experiment evaluation)')
    self.ckpt_dir         = self.parser.add_argument('--ckpt_dir'                 , type=str  , default='./checkpoints/'                 , help='directory for checkpoint files')
    self.load_task        = self.parser.add_argument('--load_task'                , type=int  , default=0                                , help='load a specified task checkpoint (0 = do not load checkpoint)')
    self.save_All         = self.parser.add_argument('--save_All'                 , type=bool , default=True                             , help='saved the model for each task (after last training iteration)')
    #-------------------------------------------------------------- EXAMPLE SLTs

    #self.parser.add_argument('--T1'                         , type=int  , default=[0, 1, 2, 3, 4]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
    #self.parser.add_argument('--T2'                         , type=int  , default=[5, 6, 7, 8, 9]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')

    #self.parser.add_argument('--T1'                         , type=int  , default=[0, 1]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
    #self.parser.add_argument('--T2'                         , type=int  , default=[2, 3]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
    #self.parser.add_argument('--T3'                         , type=int  , default=[4, 5]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
    #self.parser.add_argument('--T4'                         , type=int  , default=[6, 7]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')
    #self.parser.add_argument('--T5'                         , type=int  , default=[8, 9]       ,  help='classes for the specific task, T1, T2, ..., Dx, e.g. "--T1 0 1 2 3 4 5 6 7 8 9"')

    #---------------------------------------------------------- EXCHANGE DATASET
    #self.parser.add_argument('--change_dataset_range'       , type=str, default='[(2,3), (5,6)]'                     , help='change the dataset for all defined tasks tuple(<start>, <end>) in list')
    #self.parser.add_argument('--change_dataset'             , type=str, default='FashionMNIST')                      , help='the dataset which should be used to exchange')
    #self.parser.add_argument('--T2_change_dataset'          , type=str, default='FashionMNIST')                      , help='define a task specific exchange')
    #--------------------------------------------------------- INFLUENCE DATASET
    #self.parser.add_argument('--rotation_per_iteration'         , type=float, default=1                              , help='rotate the input data (images) of 1 degree per training iteration')
    #self.parser.add_argument('--T1_rotation'                    , type=float, default=90                             , help='a fixed rotation of the input data (images) of 90 degree for the full task')
    #self.parser.add_argument('--brightness_change_per_iteration', type=float, default=0.1                            , help='increase or decrease the brightness of the input data (images) by 0.001 each iteration (max decrease/increase= -0.5/+0.5, data clipping [-1, 1])')
    #self.parser.add_argument('--T1_brightness'                  , type=float, default=-0.5                           , help='a fixed increase or decrase the brightnes for the full task')

    #_5_work_days = '''(    # tuple(amplitude, shift, noise) # @7000 iteration
    #  (.4, .6, 0.026),  # class 0
    #  (.3, .4, 0.026),  # class 1
    #  (.3, .3, 0.026),)''' # class 2
    #self.parser.add_argument('--distribution_change_per_task', type=float, default=_5_work_days, nargs='*')
    #self.parser.add_argument('--T1_distribution'           , type=float, default=[1., .75, .5, .25, 0, 0, 0, 0, 0, 0], nargs='*')



  def __init__(self, *args, **kwargs):
    ''' construct a Experiment object '''
    self._init_parser(**kwargs)
    self.flags = self.parser.get_all_parameters()
    change_loglevel(self.log_level)
    self._init_dataset()
    self._init_variables()
    self._init_log_writer()


  def _init_dataset(self):
    ''' initialize the dataset, pre-load DAll for testing '''
    log.info('load dataset')
    self.dataset     = Dataset(**self.flags)

    self.properties  = self.dataset.properties
    self.w, self.h   = self.properties.get('dimensions') # extract image properties or use slice patch size
    self.c           = self.properties.get('num_of_channels')
    self.num_classes = self.properties.get('num_classes')
    self.flags['h']  = self.h
    self.flags['w']  = self.w
    self.flags['c']  = self.c

    if self.flags.get('DAll'):
      self.DAll                = [ int(t) for t in self.flags.get('DAll') ] # classes for DAll
      _, self.test_D_ALL, _, _ = self.dataset.get_dataset(self.DAll)


  def _init_variables(self):
    ''' init some experiment variables '''
    log.info('init variables')
    self.external_model           = None
    self.global_iteration_counter = 0
    #-------------------------------------------------------- TASK/DATASET LISTS
    taskList              = [ k for (k, _) in self.flags.items() if k.startswith('T') and (k[1:]).isnumeric() ]
    self.num_tasks        = len(taskList)
    self.tasks            = list() # stores classes for each processed task
    self.tasks_iterations = list() # stores number of (training, testing) iterations list(tuple)
    self.training_sets    = list() # stores task training datasets
    self.test_sets        = list() # stores task test datasets


  def _init_log_writer(self):
    ''' init Logging mechanism to write experiment properties and measured values to a JSON-file '''
    log.info('init logging writer')
    self.log = Logging(**self.flags)


  def _init_connection_handler(self, model):
    ''' init connection handler to distribute values/results via network socket '''
    self.connection_handler = None
    try:
      self.connection_handler = Connection_Handler(model=model, environment_params=self.flags)
      self.connection_handler.start()
    except Exception as ex:
      log.error(f'could not start connection handler: {ex}')


  def set_model(self, model, **kwargs):
    ''' set an external created model '''
    self.external_model = model


  def _create_model(self):
    ''' use external model or instantiate '''
    if self.external_model:
      log.info('use external model')
      model = self.external_model
      model.compile()
    else:
      log.info(f'instantiate {self.model_type}')
      model_module = import_module(f'DCGMM.model.{self.model_type}')
      model_class  = getattr(model_module, self.model_type)
      model        = model_class(**self.flags)

      model.build()
    self._init_connection_handler(model) # create connection handler to fetch data from model
    return model


  def feed_dict(self):
    ''' feed_dict function return a batch of data and labels for training of the current task '''
    xs, ys = next(self.training_sets[self.task])

    #xs     = self.dataset.rotate(xs, self.global_iteration_counter, self.task, **self.flags)
    #xs     = self.dataset.brighten_darken(xs, self.global_iteration_counter, self.task, **self.flags)

    return xs, ys


  def _calc_test_points(self):
    ''' calculate the training and test iterations for test points (based on the number of measuring points) '''
    test_at_iteration      = np.array_split(np.arange(self.iterations_task_all), self.measuring_points)
    test_at_iteration      = [ x[0] for x in test_at_iteration ]
    test_at_iteration     += [self.iterations_task_all - 1] # at 100%
    self.test_at_iteration = test_at_iteration

    test_all_n_batches = self.iterations_task_all // self.measuring_points
    self.test_all_n_batches = test_all_n_batches if test_all_n_batches >= 1 else 1

    if self.vis_points != 0:
      vis_all_n_batches = self.iterations_task_all // self.vis_points
      self.vis_all_n_batches = vis_all_n_batches if vis_all_n_batches >= 1 else 1


  def _load_task_dataset(self, current_task=0):
    ''' create sub-dataset and append to training and test lists '''

    def dataset_change_function(task):
      ''' change the sub-dataset only if parameter is set '''
      change_dataset_range = self.flags.get('change_dataset_range')
      if not change_dataset_range: return

      task_dataset = self.flags.get(f'T{task + 1}_change_dataset') or self.flags.get('change_dataset') # task defined or global alternative dataset
      if not task_dataset: raise Exception('Missing dataset type for dataset change function')

      for (lower, upper) in eval(change_dataset_range): # cehck if current task is in
        if lower <= task + 1 <= upper: # substitute dataset if in set, e.g. (2,3) change dataset for task T2 and T3
          log.info(f'change dataset for task {task +1} from {self.dataset.dataset_file} to {task_dataset}')
          if not hasattr(self, 'default_dataset'): self.default_dataset = self.dataset # store default dataset
          self.flags['dataset_file'] = task_dataset
          self.alternative_dataset = Dataset(**self.flags)
          self.dataset             = self.alternative_dataset
          return
      if hasattr(self, 'default_dataset'):
        log.info('change back to default dataset')
        self.dataset = self.default_dataset # reset dataset if changed
    dataset_change_function(self.task)

    if self.flags.get('load_task'): # if a checkpoint is loaded, preload the already processed datasets for full evaluation
      if self.start_task != current_task:
        log.info(f'preload datasets until task T{self.load_task + 1}')
        for self.task in range(self.start_task): self.global_iteration_counter += self._load_task_dataset(current_task=self.start_task)
        self.task += 1

    log.info(f'load dataset (T{self.task + 1})')
    task = self.flags.get(f'T{self.task + 1}')       # list of classes from command line parameter
    task = [task] if isinstance(task, int) else task # turn single tasks (int) into a list (list(str or int))

    training, testing, samples_train, samples_test = self.dataset.get_dataset(task)
    self.training_iterations, testing_iter         = samples_train // self.batch_size, samples_test // self.test_batch_size

    training, self.training_iter                   = self.dataset.change_distribution(training, self.training_iterations, self.task, **self.flags)

    self.tasks            += [task]
    self.tasks_iterations += [(self.training_iterations, testing_iter)]
    self.training_sets    += [iter(training)]
    self.test_sets        += [testing]

    self.epochs = self.epochs                      # use default epochs
    if self.flags.get(f'T{self.task + 1}_epochs'): # task specific epochs
      self.epochs = self.flags.get(f'T{self.task + 1}_epochs')
      log.info(f'use custom epochs for T{self.task + 1} (epochs={self.epochs})')

    self.iterations_task_all = int(math.ceil(self.tasks_iterations[self.task][0]) * self.epochs) # round up 0.1 -> 1

    self._calc_test_points() # calculate the position of the measurement points
    log.info(f'{"-" * 66} Task T{self.task + 1} {"-" * 66}')

    return self.iterations_task_all


  def get_task_iterations(self, task):
    ''' return (train_iter, test_iter) '''
    return self.tasks_iterations[task]


  def before_task(self, **kwargs): pass


  #----------------------------------------------------------- TASK LOOP METHODS
  def _test(self):
    ''' test
      1 epoch on: a combination/merge of all previous, current and following tasks (definition of DAll)
      1 epoch on: all inidividual previous tasks (T_1, ..., T_x-1)
      1 epoch on: current task (T_x)
    '''
    if self.iteration_task != self.test_at_iteration[0]: return
    self.test_at_iteration.pop(0)
    print ("TTTTTTEEEEEEEEEEEESSSSTTT")

    for test_task in range(-1 if hasattr(self, 'DAll') else 0, self.task + 1):                                                          # test DAll if given and all previous and current subtasks
      test_dataset    = self.test_D_ALL if test_task < 0 else self.test_sets[test_task]                                                 # get dataset to test
      task_name       = 'DAll' if test_task < 0 else f'T{test_task + 1}'                                                                # build task name
      results         = self.model.test(iter(test_dataset), task_name=task_name)                                                        # return structure dict(key=tuple(<source> (str), <metric_name> (str)): value=tuple(<raw_value> (int, float, ...), <formated value> (str))), e.g, {('L3_gmm', 'loglikelihood') : (123,43768463278436, '123.43')}
      task_classes    = '(test classes: {})'.format(','.join(map(str, self.DAll if test_task < 0 else self.tasks[test_task])))          # get tested classes

      test_step    = f'{"":>19}' if self.task != test_task else f'at step {self.iteration_task:>5} / {self.iterations_task_all - 1:>5}' # current test step
      progress     = f'({self.iteration_task / (self.iterations_task_all - 1):6.1%})' if self.task == test_task else ''                 # progress of training (only current task)

      # TODO: check with AK if this can really be killed
      """
      # TODO Fallunterscheidung raus!
      if isinstance(self.model, Aggr_Stacked_GMM): # need to loop over sub-models aswell as print overall acc
        if task_name == 'DAll': # for DAll task we evaluate on a per sample basis and calc overall accuracy
          for (model_name, source, metric_name), (acc_value_raw, acc_format) in results.items():
            test_type_value = f'{metric_name:<22}: {acc_format}'
            log.info(f'{model_name:<30} {source:<16} {task_classes:<37} {task_name:<10} {test_type_value} {test_step} {progress:<6}')
            self.log.add_eval_combination(
              keys=(metric_name, task_name, model_name),
              values=(test_task + 1, self.iteration_task, acc_value_raw, self.global_iteration_counter),
            )
        else: # for specific Tasks only evaluate sub-model log-likelihoods
          for (task, source, metric_name), (metric_value_raw, metric_value_formatted) in results.items():                                         # for each metric
            if metric_name != 'log_likelihoods':  # supress double loglikelihood printing..
              test_type_value = f'{metric_name:<22}: {metric_value_formatted}'
              model_name = f'Stacked_GMM [{task}]'
              log.info(f'{model_name:<30} {source:<15}  {task_classes:<37} {task_name:<10} {test_type_value} {test_step} {progress:<6}')
              self.log.add_eval_combination(                                                                                                  # write metrics to logging structure
                keys   = (metric_name, task_name, model_name, source),                                                                                     # is concatenated to a string and used as key
                values = (test_task + 1, self.iteration_task, metric_value_raw, self.global_iteration_counter),                                # values stored as list under the key
                )
      else:
      """
      if True:
        for (source, metric_name), (metric_value_raw, metric_value_formatted) in results.items():  # for each metric
          test_type_value = f'{metric_name:<22}: {metric_value_formatted}'  # build current metric string
          log.info(f'{source:<30}  {task_classes:<37} {task_name:<10} {test_type_value} {test_step} {progress:<6}')
          self.log.add_eval_combination(  # write metrics to logging structure
          keys = (metric_name, task_name, source),  # is concatenated to a string and used as key
          values = (test_task + 1, self.iteration_task, metric_value_raw,
                    self.global_iteration_counter),  # values stored as list under the key
          )
    log.info('-' * 140)
    #self.model.vars_to_numpy(prefix=self.results_dir)

  def extra_info(self, xs, ys, **kwargs):
    return {}

  # TODO task and task_iteration as parameters!
  # TODO discuss this
  # TODO maybe add some kind of parameter for different training funcs
  def _train_step(self, **kwargs):
    xs, ys = self.feed_dict()
    extra = self.extra_info(xs, ys, **kwargs) 
    kwargs.update(extra) 
    #train_eval = self.model.train_step_mask(xs=xs, ys=ys, mask_alpha=0.9)
    train_eval = self.model.train_step(xs=xs, ys=ys, task=f'T{str(self.task+1)}', **kwargs)
    # if model._train_step chooses not to return anything --> do nothing
    # if it returns a dict(same format as for testing) --> the content is logged as during testing
    if train_eval is not None:
      task_name = "T" + str(self.task)
      for (source, metric_name), (metric_value_raw, metric_value_formatted) in train_eval.items():
        self.log.add_eval_combination(                                                                  
          keys   = (metric_name, task_name, source),                                                    
          values = (self.task, self.iteration_task, metric_value_raw, self.global_iteration_counter))


  def _share_variables(self):
    ''' transmit model/layer variables by network (socket) '''
    if self.vis_points == 0                             : return
    if not self.connection_handler                      : return
    if self.iteration_task % self.vis_all_n_batches != 0: return
    if not self.connection_handler.has_clients()        : return

    self.connection_handler.fetch_and_send(
      self.model.layers                           ,
      iteration_glob=self.global_iteration_counter,
      )
  #------------------------------------------------------ TASK LOOP METHODS ENDS


  def after_task(self, **kwargs) : pass


  def train(self):
    ''' generic training loop for all Experiments '''
    self.model      = self._create_model()
    self.start_task = self.model.load()                               # if exists and is to be loaded load checkpoint of the model

    for self.task in range(self.start_task, self.num_tasks):          # loop over all tasks
      self._load_task_dataset()                                       # load dataset for current task and previous tasks if a model is loaded

      self.before_task()
      for self.iteration_task in range(self.iterations_task_all):     # loop for one task
        self._test()
        self._train_step()
        self._share_variables()                                       # trigger API to transmit model variables
        self.global_iteration_counter += 1

      self.after_task()
      self.model.save(current_task=self.task)                         # if parameter is set, create a checkpoint of the model

    self.log.write_to_file()

