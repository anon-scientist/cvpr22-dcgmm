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

# TODO default-parameter Problematik in init_parser
import sys
import numpy            as np
from .                  import Experiment
from DCGMM.utils        import log

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class Experiment_GMM(Experiment):
  ''' defines a single GMM experiment with different tasks and parameters '''

  def _init_parser(self, **kwargs):
    Experiment._init_parser(self, **kwargs)
    #---------------------------------------------------------- MODEL CLASS NAME
    self.model_type                   = self.parser.add_argument('--model_type'                 , type=str    , default='Stacked_GMM' , help='class name in model sub-dirrectory to instantiate')
    #------------------------------------------------------------- GMM PARAMETER
    self.reset_factor                 = self.parser.add_argument('--reset_factor'               , type=float  , default=-1.0          , help='for new tasks trigger the start values of annealing controll (somSigma and eps) are reduced by the factor (for all tasks) 1=reset to start value, 1=reset to last possible value, -1=no reset')

    self.masked_training              = self.parser.add_argument('--masked_training'                , type=eval     , default=False           , help='switch masked training on/off')
    self.wrong_level                  = self.parser.add_argument('--wrong_level'                  , type=float    , default=1.                 , help='mask entry for incorrect classifications')
    self.right_level                  = self.parser.add_argument('--right_level'                  , type=float    , default=0.                 , help='mask entry for correct classifications')

    self.perform_inpainting            = self.parser.add_argument('--perform_inpainting'                , type=eval     , default="False"           , help='switch inpainting on/off')
    self.perform_variant_generation   = self.parser.add_argument('--perform_variant_generation' , type=eval   , default="False"       , help='switch vargen on/off')
    self.perform_sampling             = self.parser.add_argument('--perform_sampling' , type=eval   , default=False       , help='switch pre/post sampling on/off')

    self.variant_gmm_root             = self.parser.add_argument('--variant_gmm_root'      , type=int    , default=-1          , help='generate variants from which gmm layer downwards?')
    self.cond_sampling_classes        = self.parser.add_argument('--cond_sampling_classes'      , type=int    , default=None          , help='condsampling for which classes?')
    if type(self.cond_sampling_classes) is type(1): self.cond_sampling_classes = [self.cond_sampling_classes]
    self.sampling_layer               = self.parser.add_argument('--sampling_layer'             , type=int    , default=-1            , help='which layer is used to start sampling?')
    self.nr_sampling_batches          = self.parser.add_argument('--nr_sampling_batches'             , type=int    , default=1            , help='how many rounds of sampling to perform??')


  def _reset_factor(self):
    try   : reset_factor = self.flags[f'T{self.task}_reset_factor']   # use task specific reset factor
    except: reset_factor = self.reset_factor                          # use general reset factor
    log.debug(f'use reset factor {reset_factor}')
    return reset_factor


  def _train_step(self, **kwargs):
    perc = self.iteration_task / (self.iterations_task_all + 1.)
    self.model.set_parameters(percentage_task_done = perc )
    Experiment._train_step(self, **kwargs)

  def do_sampling(self, prefix="sampling_"):
    if self.perform_sampling == False: return ;
    for i in range(0,self.nr_sampling_batches):
      T = self.cond_sampling_classes 
      print ("T IS ", T, "sampling layer is", self.sampling_layer)
      batch = None

      if T is not None and self.sampling_layer  == len(self.model.layers)-1:
        topdown_signal,one_hot = self.model.construct_topdown_for_classifier(self.num_classes, 0.95, T)      
        print ("TOPDIOWN SHAPE=",topdown_signal.shape)
        batch = self.model.sample_one_batch(topdown=topdown_signal)
        np.save(self.results_dir+"/"+self.exp_id+"_"+str(i)+"_samples.npy", batch.numpy()) ;
        np.save(self.results_dir+"/"+self.exp_id+"_"+str(i)+"_labels.npy", one_hot.numpy()) ;
      elif T is not None and self.sampling_layer < len(self.model.layers):
        batch = self.model.sample_one_batch(topdown=None, last_layer_index=self.sampling_layer)
        np.save(self.results_dir+"/"+self.exp_id+"_"+str(i)+"_samples.npy", batch.numpy()) ;
        topdown_signal,one_hot = self.model.construct_topdown_for_classifier(self.num_classes, 0.95, T)      
        np.save(self.results_dir+"/"+self.exp_id+"_"+str(i)+"_labels.npy", one_hot.numpy()) ;
      else:
        batch = self.model.sample_one_batch(topdown=None, last_layer_index=self.sampling_layer)
        np.save(self.results_dir+"/"+self.exp_id+"_"+str(i)+"_samples.npy", batch.numpy()) ;

      #sampled_loss = self.model.get_outlier_loss(batch) ;
      #print ("Sampled_loss=", sampled_loss.numpy())
      print ("SAAAAAAAMPLING", batch.shape)
      self.model.samples_2_numpy(prefix=self.results_dir+"/"+self.exp_id+"_"+prefix, sampled=batch)



  def before_task(self, **kwargs):
    #sampling
    print("HERE>>>>>>>>>>>>>>>>>>>>>>>>>>>", self.task);
    self.do_sampling()
    if self.task >= 1:
      self.model.reset(reset_factor=self._reset_factor()) # new task trigger with reset factor
      print ("VARIANTS!!!", self.perform_variant_generation) ;
      print ("INPAINTING!!!", self.perform_inpainting) ;
      if self.perform_variant_generation == True:
        #xs, ys = self.feed_dict()
        xs, ys = next(self.training_sets[self.task-1])

        self.model.samples_2_numpy(prefix='data_', sampled=xs)
        self.model.samples_2_numpy(prefix='variants_', sampled=self.model.do_variant_generation(xs, bu_limit_layer = self.variant_gmm_root))

      if self.perform_inpainting == True:
        print ("-------------------------IIIIIIIIIIIIN") ;
        #xs, ys = self.feed_dict()
        xs, ys = next(self.training_sets[self.task-1])

        n,h,w,c = xs.shape
        mask = np.ones([n,h,w,c]) ;
        mask[:,:,w//2:,:] = 0

        variants = self.model.do_variant_generation(xs * mask, bu_limit_layer = self.variant_gmm_root)
        result = variants *(1. - mask) + xs * mask ;

        self.model.samples_2_numpy(prefix='data_', sampled = xs*mask)
        self.model.samples_2_numpy(prefix='variants_', sampled = result)


  def after_task(self, **kwargs):
    self.do_sampling()


if __name__ == '__main__':
  Experiment_GMM().train()
