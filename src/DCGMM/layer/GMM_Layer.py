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

import math
import numpy      as np
import tensorflow as tf

from .regularizer import Regularizer_Method as RM
from .            import Layer
from DCGMM.utils  import log
from DCGMM.metric import Metric
from importlib    import import_module

class Mode:
  DIAG   = 'diag'
  FULL   = 'full'

class Energy:
  LOGLIK     = 'loglik'
  MC         = 'mc'


class GMM_Layer(Layer):
  ''' a GMM layer '''

  def __init__(self, input=None, **kwargs):
    '''
    @param input_: input tensor of the previous layer
    @param kwargs: parameter dictionary with all parameters (all layers),
           layer specific parameters are in self.layer_params (extracted in base class)
    '''
    Layer.__init__(self, input=input, **kwargs)
    self.kwargs                = kwargs

    self.batch_size            = self.parser.add_argument('--batch_size'         , type=int  , default=100                                  , help='used batch size')
    self.name                  = self.parser.add_argument('--name'               , type=str  , default=f'{self.prefix}gmm'                  , help='name of the gmm layer')
    self.K                     = self.parser.add_argument('--K'                  , type=int  , default=10 ** 2                              , help='number of gmm components')
    self.n                     = int(math.sqrt(self.K))
    self.mode                  = self.parser.add_argument('--mode'               , type=str  , default=Mode.DIAG                            , help='"diag" or "full" type of used covariance matrix', choises=['diag'])
    self.sampling_divisor      = self.parser.add_argument('--sampling_divisor'   , type=float, default=1.                                   , help='divide stddev by this factor in sampling')
    self.muInit                = self.parser.add_argument('--muInit'             , type=float, default=0.01                                 , help='initialization value of prototypes')
    self.sigmaUpperBound       = self.parser.add_argument('--sigmaUpperBound'    , type=float, default=20                                   , help='the upper bound for clipping sigmas')
    self.somSigma0             = self.parser.add_argument('--somSigma0'          , type=float, default=0.25 * math.sqrt(2 * self.K)         , help='only use auto initialization of somSigma0')
    self.somSigmaInf           = self.parser.add_argument('--somSigmaInf'        , type=float, default=0.01                                 , help='smalles sigma value for regularization')
    self.eps0                  = self.parser.add_argument('--eps0'               , type=float, default=0.05                                 , help='start epsilon value (initial learning rate)')
    self.epsInf                = self.parser.add_argument('--epsInf'             , type=float, default=0.049                                , help='smalles epsilon value (learn rate) for regularization')
    self.energy                = self.parser.add_argument('--energy'             , type=str  , default=Energy.MC                            , help='"loglik" (standard log-likelihood loss using log-sum-exp-trick) or "mc" (MC approximation) the used energy function')
    self.lambda_pi             = self.parser.add_argument('--lambda_pi'          , type=float, default=0.                                   , help='adaption factor for the pis')
    self.lambda_mu             = self.parser.add_argument('--lambda_mu'          , type=float, default=1.                                   , help='factor for the mus')
    self.lambda_sigma          = self.parser.add_argument('--lambda_sigma'       , type=float, default=.5                                   , help='factor for the sigmas')
    self.wait                  = self.parser.add_argument('--wait'               , type=float, default=0                                    , help='determines the pefcentage of a sub-tasks time a layer is supposed to be inert (not trained). Useful for higher DCGMM layers')
    self.return_loss           = self.parser.add_argument('--return_loss'        , type=str  , default='loglikelihood'                      , help='the provided loss tensor for other layers')
    self.regularizer           = self.parser.add_argument('--regularizer'        , type=str  , default=RM.SINGLE_EXP                        , help='"single_exp" (sliding average control) or "time_decay" (depending on iterations) regularizer type')
    self.enable_bmu_counter    = self.parser.add_argument('--enable_bmu_counter' , type=eval , default=False                                , help='enables the BMU counter')
    self.convMode              = self.parser.add_argument('--convMode'           , type=eval , default=True, choices=[True,False]           , help='if true, one gmm layer is used for input else, for each input patch one gmm layer is created')
    self.loss_factor           = self.parser.add_argument('--loss_factor'        , type=float, default=1.                                   , help='factor for multiplying resulting layer loss')

    # sampling parameters
    self.sampling_active       = self.parser.add_argument('--sampling_active'    , type=eval , default=False                                , help='TODO')
    self.use_pis               = self.parser.add_argument('--use_pis'            , type=eval , default=False                                , help='use pis when sampling without topdown signal')
    self.sampling_batch_size   = self.parser.add_argument('--sampling_batch_size', type=int  , default=100                                  , help='sampling batch size')
    self.sampling_S            = self.parser.add_argument('--sampling_S'         , type=int  , default=10                                   , help='select best x prototypes for sampling')
    self.sampling_P            = self.parser.add_argument('--sampling_P'         , type=int  , default=1                                    , help='power to raise topdown priors to. 1--> no_op')
    self.sampling_I            = self.parser.add_argument('--sampling_I'         , type=int  , default=-1                                   , help='index of selected component')

    # evaluation metrics
    default_metrics            = list()
    default_metrics           += ['log_likelihood', 'log_likelihoods'] # same as 'loss'
    #default_metrics           += ['davies_bouldin_score', 'dunn_index'] # these metrics are very compute intensive
    self.metrics               = self.parser.add_argument('--metrics'            , type=str  , default=default_metrics                      , help='evaluation metrics: log_likelihood, davies_bouldin_score, dunn_index')
    self.metric                = Metric(self)

    input_shape                = self.prev.get_shape()
    self.batch_size            = input_shape[0]
    self.h_out                 = input_shape[1]
    self.w_out                 = input_shape[2]
    self.c_in                  = input_shape[3]
    self.c_out                 = self.K
    log.debug(f'Output shape is {self.batch_size},{self.h_out},{self.w_out}, {self.K}')

    self.percentage_task_done = 1.0   # for implementing 'wait' behavior


  def get_shape(self):
    return [self.batch_size, self.h_out, self.w_out, self.c_out]


  def get_layer_variables(self, all_=False):
    ''' This saves all trainable parameters of the current layer into a dictionary to return '''
    variables = {        # all trainable variables
      f'{self.prefix}mus'   : self.mus   ,
      f'{self.prefix}sigmas': self.sigmas,
      f'{self.prefix}pis'   : self.pis   ,
      }
    if all_:
      variables.update({ # add other variables to load/create checkpoints
      f'{self.prefix}somSigma': self.tfSomSigma,
      f'{self.prefix}tfEps'   : self.tfEps     ,
      f'{self.prefix}convMasks'   : self.conv_masks
        })
    return variables


  def is_trainable(self): return True


  def set_parameters(self, **kwargs):
    self.percentage_task_done = kwargs.get('percentage_task_done', 1.0)


  def evaluate(self, logits, **kwargs):
    evaluate_dict = {
      'log_likelihoods': self.loglikelihood_full                ,
      }
    return evaluate_dict


  def compile(self):
    ''' Initialization of all TF variables of this layer (variables, constants).
    Gets called upon layer init:
      Variables for time-varying factors: could be varied but kept to 1.0 here
      - covariance (sigma) factor
      - centroids (mu) factor
      - weights (pi) factor (selection probability)
    '''
    self.lambda_sigma_factor = self.variable(self.lambda_sigma, shape=[], name='lambda_sigma')
    self.lambda_mu_factor    = self.variable(self.lambda_mu   , shape=[], name='lambda_mu')
    self.lambda_pi_factor    = self.variable(self.lambda_pi   , shape=[], name='lambda_pi')

    # TODO checken ob das sinnvoll ist: für convMode = False muss der loss ja nicht gemean'ed werden --> korrigiere mit Faktor beim update
    self.loss_mult           = self.h_out*self.w_out 

    # define shapes
    pis_shape                = [1, self.h_out, self.w_out, self.K]
    mus_shape                = [1, self.h_out, self.w_out, self.K, self.c_in]
    sigmas_shape             = None
    D_shape                  = [1, self.h_out, self.w_out, self.K, self.c_in]

    # full covariance matrices are initialized to diagonal ones with diagonal entries given by sigmaUpperBound
    if self.mode == Mode.DIAG:
      sigmas_shape    = [1, self.h_out, self.w_out, self.K, self.c_in]


    log.debug(f"convMode={self.convMode}") ;
    if self.convMode:
      sigmas_shape[1] = sigmas_shape[2] = 1
      D_shape[1]      = D_shape[2]      = 1
      pis_shape[1]    = pis_shape[2]    = 1
      mus_shape[1]    = mus_shape[2]    = 1
      self.loss_mult  = 1.0 ;

    # var initializers
    init_pi           = self.constant_initializer(1. / self.K)
    init_eps          = self.constant_initializer(self.eps0)
    init_somSigma     = self.constant_initializer(self.somSigma0)
    init_rand_mu      = tf.initializers.RandomUniform(-self.muInit, +self.muInit)


    self.tfEps        = self.variable(initial_value=init_eps(shape=[]),            shape=[],        name='eps')
    self.tfSomSigma   = self.variable(initial_value=init_somSigma(shape=[]),       shape=[],        name='somSigma')
    # the raw pis, before used they are passed through a softmax
    self.pis          = self.variable(initial_value=init_pi(shape=pis_shape),      shape=pis_shape, name='pis')
    self.mus          = self.variable(initial_value=init_rand_mu(shape=mus_shape), shape=mus_shape, name='mus')
    self.D            = tf.constant(1.0)
    if self.mode == Mode.DIAG:
      init_sigma      = self.constant_initializer(math.sqrt(self.sigmaUpperBound))
      self.sigmas     = self.variable(initial_value=init_sigma(shape=sigmas_shape), shape=sigmas_shape, name='sigmas')

    # constant term in log probabilities
    self.const_         = tf.constant(-self.c_in / 2. * tf.math.log(2. * math.pi))                                                           # --> N,pY,pX,K


    def prepare_annealing():
      ''' generate structures for efficiently computing the time-varying smoothing filter for the annealing process '''
      shift          = +1 if self.n % 2 == 1 else 0
      oneRow         = np.roll(np.arange(-self.n // 2 + shift, self.n // 2 + shift, dtype=self.dtype_np_float), self.n // 2 + shift).reshape(self.n)
      npxGrid        = np.stack(self.n * [oneRow], axis=0)
      npyGrid        = np.stack(self.n * [oneRow], axis=1)
      npGrid         = np.array([ np.roll(npxGrid, x_roll, axis=1) ** 2 + np.roll(npyGrid, y_roll, axis=0) ** 2 for y_roll in range(self.n) for x_roll in range(self.n) ])
      self.xyGrid    = self.constant(npGrid.reshape(1, 1, 1, self.K, self.K))
    prepare_annealing()

    self.conv_masks       = tf.Variable(initial_value=self.xyGrid)
    self.last_som_sigma   = float('inf')  # to remember somSigma value from last iteration. If changed --> recompute filters

    reg_class             = getattr(import_module(f'DCGMM.layer.regularizer.{self.regularizer}'), self.regularizer)
    self.reg              = reg_class(**{**vars(self), **self.kwargs}) # instantiate given regularizer class object from "layer.regularizer" package
    #self.regs             = [reg_class(**{**vars(self), **self.kwargs}) for i in range(int(self.loss_mult))]
 
    if self.enable_bmu_counter:
      self.init_bmu_counter = self.constant_initializer(0)
      self.bmu_counter      = self.variable(initial_value=self.init_bmu_counter(shape=[self.K]), shape=[self.K], dtype=tf.int32, trainable=False, name='bmu_counter')


  def recompute_smoothing_filters(self, convMaskVar):
    ''' if regularizer has decreased sigma, recompute filter variable. Otherwise do nothing '''
    if (self.last_som_sigma > self.tfSomSigma):
      convMaskVar.assign(tf.exp(-self.xyGrid / (2.0 * self.tfSomSigma ** 2.0)))
      convMaskVar.assign(convMaskVar / (tf.reduce_sum(convMaskVar, axis=4, keepdims=True)))
      self.last_som_sigma = self.tfSomSigma.numpy()


  @tf.function(autograph=False)
  def forward(self, input_tensor, extra_inputs = None):
    ''' raw input --> log-scores, i.e. log (p_k p_k) --> N,pY,pX,K. input_tensor is the mini-batch, runs eagerly '''
    diffs     = tf.expand_dims(input_tensor, 3) - self.mus                                                            # (N, pY,pX, 1, D)  - (1,pY,pX,K,D) --> (N,pY,pX,K,D)

    if self.mode == Mode.DIAG:
      log_det = tf.reduce_sum(tf.math.log(self.sigmas), axis=4, keepdims=False)                                       # sum(1,pY,pX,K,D,axis=4) --> 1,pY,pX,K
      sqDiffs = diffs ** 2.0                                                                                          # --> N,pY,pX,K,D
      log_exp = -0.5 * tf.reduce_sum(sqDiffs * (self.sigmas ** 2.), axis=4)                                           # sum(N,pY,pX,K,D * N,pY,pX,K,D,axis=4) --> N,pY,pX,K
    else:
      raise Exception("Non-diag cmatrix not supported. Try MFA!")

    log_probs = (log_det + log_exp)                                                                                   # --> N,pY,pX,K
    exp_pis   = tf.exp(self.pis)                                                                                      # --> 1,1,1,K # obtain real pi values by softmax over the raw pis thus, the real pis are always positive and normalized
    real_pis  = exp_pis / tf.reduce_sum(exp_pis)

    self.log_scores = tf.math.log(real_pis) + log_probs                                                               # --> N,pY,pX,K

    return self.log_scores


  @tf.function(autograph=False)
  def get_output(self, log_scores):
    ''' produce output to next layer, here: log-scores to responsibility '''
    max_logs    = tf.reduce_max(log_scores, axis=3, keepdims=True)                # --> N,pY,pX,1
    norm_scores = tf.exp(log_scores - max_logs)                                   # --> N,pY,pX,K
    self.resp   = norm_scores / tf.reduce_sum(norm_scores, axis=3, keepdims=True) # --> N,pY,pX,K
    return self.resp

  def pre_train_step(self):
    # only recomputes if necessary
    self.recompute_smoothing_filters(self.conv_masks)

    # only learn sth when we are allowed to, ie waiting period is over
    self.lambda_pi_factor.assign(0 if self.percentage_task_done < self.wait else self.lambda_pi)
    self.lambda_mu_factor.assign(0 if self.percentage_task_done < self.wait else self.lambda_mu)
    self.lambda_sigma_factor.assign(0 if self.percentage_task_done < self.wait else self.lambda_sigma)

  def post_train_step(self):
    #last_log_likelihood = tf.reduce_mean(self.loglikelihood_full) # batch/patch mean of loglik
    if self.percentage_task_done >= self.wait:
      if True:
        last_log_likelihood = tf.reduce_mean(self.raw_loss) # batch/patch mean of loglik
        self.reg.add(last_log_likelihood).check_limit()
      """
      else:
        last_log_likelihood = tf.reduce_mean(self.raw_loss,axis=0) # batch/patch mean of loglik
        self.reg.add(last_log_likelihood).check_limit()
      """

    


  def loss(self, log_scores, **kwargs):
    ''' log-scores to loss, with batch and patch indices: N,pY,pX, this is to optimize
     loss is returned per sample!!
    '''
    self.loglikelihood_full = self.graph_loss(log_scores) ;
    return self.loglikelihood_full

  
  @tf.function(autograph=False)
  def graph_loss_norm(self, log_scores):
      ''' test with nomalized loss, not successful so far '''
      log_piprobs        = tf.expand_dims(log_scores, axis=3)                   # --> expand3(1,pY,pX,K + N,pY,pX,K ) --> N,pY,pX,1,K
      conv_log_probs     = tf.reduce_sum(log_piprobs * self.conv_masks, axis=4) # sum4(N,pY,pX,1,K * 1,1,1,K,K) = sum4(N,pY,pX,1,K * 1,1,1,K,K) --> N,pY,pX,K
      max_logs    = tf.reduce_max(conv_log_probs, axis=3, keepdims=True)                # --> N,pY,pX,1
      norm_scores = tf.exp(conv_log_probs - max_logs)                                   # --> N,pY,pX,K
      resp   = norm_scores / tf.reduce_sum(norm_scores, axis=3, keepdims=True) # --> N,pY,pX,K
      loglikelihood_full = tf.math.log(tf.reduce_max(resp,axis=3))  # --> N,pY,pX
      #cluster_ids   = tf.argmax(tf.squeeze(conv_log_probs), axis=1)
      return loglikelihood_full


  @tf.function(autograph=False)
  def graph_loss(self, log_scores):
      log_piprobs        = tf.expand_dims(log_scores, axis=3)                   # --> expand3(1,pY,pX,K + N,pY,pX,K ) --> N,pY,pX,1,K
      conv_log_probs     = tf.reduce_sum(log_piprobs * self.conv_masks, axis=4) # sum4(N,pY,pX,1,K * 1,1,1,K,K) = sum4(N,pY,pX,1,K * 1,1,1,K,K) --> N,pY,pX,K
      loglikelihood_full = tf.reduce_max(conv_log_probs, axis=3) + self.const_  # --> N,pY,pX

      '''
      self.cluster_ids   = tf.argmax(tf.squeeze(conv_log_probs), axis=1)
      if self.enable_bmu_counter:
        self.bmu_counter.assign_add(tf.histogram_fixed_width(
          values      = tf.cast(self.cluster_ids, tf.float32),
          value_range = [0.0, self.K]  ,
          nbins       = self.K         ,
          dtype       = tf.dtypes.int32,
          ))
      '''
      return loglikelihood_full


  def update_with_grad(self, grads):

    self.pis.assign_add(   self.lambda_pi_factor    * self.tfEps * grads[self.prefix + 'pis'] * self.loss_mult)
    self.mus.assign_add(   self.lambda_mu_factor    * self.tfEps * grads[self.prefix + 'mus'] * self.loss_mult)
    self.sigmas.assign_add(self.lambda_sigma_factor * self.tfEps * grads[self.prefix + 'sigmas'] * self.loss_mult)

    if self.mode == Mode.DIAG: # sigma clipping for diag!
      sigma_limit = math.sqrt(self.sigmaUpperBound)
      self.sigmas.assign(tf.clip_by_value(self.sigmas, -sigma_limit, sigma_limit))


  def backwards(self, topdown, *args, **kwargs):
    ''' Backward pass: sampling operator (create a batch of samples from the prototypes)
      - lower layer: N, h, w, cIn
      - upperLayer : N, h, w, cOut
      - input      : fromUpperLayer = N, h, w, cOut
      - output     : N, h, w, cIn
    GMM components are picked over probabilities pi (e.g. pick 3 components: [0.8, 0.1, 0.1]
    First component is contained in 80% of all samples, second in 10%, etc...
    This procedure is hierarchical, samples are constructed from typical outputs of the underlying layer,
    Outputs gets passed until back to first layer of the model.
    '''

    N, h, w, cOut = self.sampling_batch_size, self.h_out, self.w_out, self.K # input fromUpperLayer
    cIn           = int(self.prev.get_shape()[3])                            # output (to lower layer)

    # GMM_Layer can be top-level layer, so we need to include the case without control signal
    if topdown is None:
      if self.use_pis == False:
        topdown = tf.ones([N,h,w,cOut])
      else:
        e  = tf.exp(self.pis)
        sm = e / (tf.reduce_sum(e))
        topdown = tf.stack([sm for _ in range(0,self.sampling_batch_size)])

    self.sampling_placeholder = topdown

    log.debug(f'sampling_S {self.sampling_S}')

    selectionTensor = None
    if self.sampling_I == -1: # I = -1 --> top-S-sampling
      powered         = tf.pow(tf.clip_by_value(self.sampling_placeholder, 0.000001, 11000.), self.sampling_P)
      if self.sampling_S > 0:   # S > 0: select the top S topdown values # default: top-S sampling: pass on just the P strongest probabilities. We can erase the sub-leading ones, tf.multinomial will automatically re-normalize
        sortedTensor    = tf.sort(powered, axis=3)
        selectionTensor = powered * tf.cast(tf.greater_equal(powered, tf.expand_dims(sortedTensor[..., -self.sampling_S], 3)), self.dtype_tf_float)
        selectionTensor = tf.reshape(selectionTensor, (-1, cOut))
      else: # S <= 0:  select from all topdown values
        selectionTensor = tf.reshape(powered, (-1, cOut))

    if   self.sampling_I == -2: selectorsTensor = np.arange(0, N * h * w) % cOut                                                                         # I=-2: cycle through selected components
    elif self.sampling_I == -1: selectorsTensor = tf.reshape(tf.compat.v1.random.categorical(logits=tf.math.log(selectionTensor), num_samples=1), (-1,)) # I = -1: top-S sampling  # --> N * _w * _h
    else                      : selectorsTensor = tf.ones(shape=(N * h * w), dtype=self.dtype_tf_int) * self.sampling_I                                  # I >= 0:directly select components to sample from # --> N * _w * _h

    # sampling mask: zero selected samples prototypes because sampling_placehiolder entries were < 0
    sampling_mask = tf.expand_dims(tf.cast(tf.greater(tf.reduce_sum(self.sampling_placeholder,axis=3), 0.), self.dtype_tf_float) , 3) ; # --> ?,hIn,wIn,1
    print("MEANS=", tf.reduce_mean(sampling_mask, axis=(1,2,3))) 

    # select mus and sigmas according to the post-processed topdown tensor need to distinguish between convMode (only one set of mus/sigmas for all RFs) and ind mode (separate mus/sigmas forall RFs)
    #  selectorsTensor has indices of centroids, 0,...,K
    # We select them either from mus[0,0,0] since there is only a single set of mus in convMode
    # ... or from  all independnet mus, in which case we have to do some unplesant stuff
    if self.convMode == True:
      selectedMeansTensor  = tf.gather(self.mus[0, 0, 0]   , selectorsTensor, axis=0, batch_dims=0)   # select only the prototypes --> N , D
      selectedSigmasTensor = tf.gather(self.sigmas[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)   # --> N, ?
    else:
      # TODO: this is a horribly memory-inefficient way to select prototypes and sigmas
      # not very bad since mem is freed immediately afterwards, but if we should have time ... fix it!
      musTmp = tf.reshape(tf.stack([self.mus for i in range(0,N)],axis=0),(N*h*w,self.K,-1))  # .--.-> Nhw,K,D
      sigmasTmp = tf.reshape(tf.stack([self.sigmas for i in range(0,N)],axis=0),(N*h*w,self.K,-1))  # .--.-> Nhw,K,D
      selectedMeansTensor  = tf.gather(musTmp   , selectorsTensor, axis=1, batch_dims=1)   # select only the prototypes --> N , D
      selectedSigmasTensor = tf.gather(sigmasTmp, selectorsTensor, axis=1, batch_dims=1)   # --> N, ?

    if self.mode == Mode.DIAG:
      sigma_limit       = math.sqrt(self.sigmaUpperBound)
      mask              = tf.cast(tf.less(selectedSigmasTensor, sigma_limit), self.dtype_tf_float)
      covariancesTensor =  ((1. / (selectedSigmasTensor + 0.00001)))  * mask
      mean              = tf.cast(tf.reshape(selectedMeansTensor, (-1,)), self.dtype_tf_float)
      stddev            = tf.cast(tf.reshape(covariancesTensor  , (-1,))  /self.sampling_divisor, self.dtype_tf_float)
      shape             = [N * w * h * cIn]
      mvn_tensor        = tf.random.normal(
        shape  = shape ,
        mean   = mean  ,
        stddev = stddev,
        dtype  = self.dtype_tf_float
      )

    sampling_op = tf.reshape(mvn_tensor, (N, h, w, cIn))

    #return sampling_op 
    # mask indicates where topdown was uniformly negative, indicating that
    # a pooling layer in sparse mode sampled a column that is to be ignored.
    # Where mask (h,w,0) == 0, we set a whole channel to 0
    # this is sth the folding layer below can detect
    return sampling_op*sampling_mask ;


  def post_test_step(self, results, xs, ys=None, **kwargs):
    loss                  = results.get('loss')
    sample_loglikelihoods = results.get('log_likelihoods')  # log-likelihood for each sample
    responsibilities      = results.get('responsibilities') # responsibilities for each sample
    cluster_ids           = results.get('cluster_ids')
    xs                    = tf.reshape(xs, [len(xs), -1])

    # TODO GMM_Layer should not hard-code eval measures that are used. That should come from parameters!!
    metric_values      = self.metric.eval(
      dict             = True            ,      # return a dictionary with metric values
      X                = xs              ,      # used by: davies bouldin score (max. 5000 because davis_boudlin score is slow)
      points           = xs              ,      # used by: dunn index (max. 5000 because dunn_index is slow)
      labels           = cluster_ids     ,      # used by: clustering metrics (dunn index, davies bouldin score) (max. 5000 because dunn_index/davis_boudlin score is slow)
      log_likelihood   = tf.reduce_mean(sample_loglikelihoods) ,      # used by: log-likelihood
      log_likelihoods  = sample_loglikelihoods, # used by: Aggr_Stacked_GMM
      num_batches      = ys              ,      # used by: outlier batch
      num_samples      = len(xs)         ,      # used by: outlier samples
      loss             = loss * self.loss_factor,
      responsibilities = responsibilities,
      )
    return metric_values


  def reset_layer(self, **kwargs):
    ''' semantics of reset_factor: -1 --> no_op, sigma --> sigma * (1 - reset_factor) '''

    if self.enable_bmu_counter:
      log.debug(f'reset BMU counter for {self.name}')
      self.bmu_counter.assign(tf.zeros([self.K], dtype=tf.int32)) # TODO: find a better way to reset a variable to its initializer

    reset_factor = kwargs.get('reset_factor', -1)
    if reset_factor  == -1: return
    reset_somSigma = self.somSigma0 - ((self.somSigma0 - self.somSigmaInf) * (1 - reset_factor))
    self.last_som_sigma = 1000000000000.

    log.debug(f'reset {self.name} (factor={reset_factor}): reset somSigma={reset_somSigma}')
    self.tfSomSigma.assign(reset_somSigma)

    if hasattr(self, 'activate_sigma_training'):
      self.reg.currentSigma = reset_somSigma
      #self.lambda_sigma     = 0
      #del(self.activate_sigma_training) # disable sigma training

    if reset_factor == 1.0: # full reset
      self.pis.assign(self.init_pi(shape=self.pis_shape))
      self.mus.assign(self.init_rand_mu(shape=self.mus_shape))
      if self.mode == Mode.DIAG:
        self.sigmas.assign(self.constant_initializer(math.sqrt(self.sigmaUpperBound)))


  def share_variables(self, **kwargs):
    iteration_glob = kwargs.get('iteration_glob')
    # create generators for data extraction
    mus            = self.mus.numpy()
    pis            = self.pis.numpy()
    sigmas         = self.sigmas.numpy()
    eps            = self.tfEps.numpy()
    # conv_masks     = self.convMasks, w=4, feed_dict={} # FIXME: rework to TF2
    if self.enable_bmu_counter:
      bmu_counter  = self.bmu_counter.numpy()
    data_dict      = dict()
    h_out          = 1 if self.convMode else self.h_out
    w_out          = 1 if self.convMode else self.w_out

    if kwargs.get('mus,pis' , False): data_dict['mus,pis']  = [ dict(mus         = mus[i]                  , pis=pis[i], iteration=iteration_glob) for i, _ in enumerate(range(h_out * w_out)) ]
    if kwargs.get('sigmas'  , False): data_dict['sigmas']   = [ dict(sigmas      = sigmas[i]               ,             iteration=iteration_glob) for i, _ in enumerate(range(h_out * w_out)) ]
    if kwargs.get('eps'     , False): data_dict['eps']      = [ dict(eps         = eps[0]                  ,             iteration=iteration_glob)                                             ] # TODO: implement a visualizer
    #if kwargs.get('convMask', False): data_dict['convMask'] = [ dict(convMask    = conv_masks              ,             iteration=iteration_glob)                                             ] # TODO: implement a visualizer
    if kwargs.get('reg'     , False): data_dict['reg']      = [ dict(reg         = self.reg.vis()          ,             iteration=iteration_glob)                                             ] # TODO: implement a visualizer
    if kwargs.get('loglik'  , False): data_dict['loglik']   = [ dict(loglik      = self.last_log_likelihood,             iteration=iteration_glob)                                             ] # TODO: implement a visualizer
    if self.enable_bmu_counter:
      if kwargs.get('bmu'     , False): data_dict['bmu']      = [ dict(bmu_counter = bmu_counter             ,             iteration=iteration_glob)                                             ]
    return data_dict


  def fetch_variables_conf(self):
    ''' fetch variables values from layer for to  '''
    variables = {
      'K'       : self.K                ,
      'n'       : self.n                ,
      'convMode': self.convMode         ,
      'h_out'   : self.h_out            ,
      'w_out'   : self.w_out            ,
      'c_in'    : self.c_in             ,
      'width'   : self.prev.patch_width ,
      'height'  : self.prev.patch_height,
      'channels': self.prev.c_in        ,
      }
    return variables


