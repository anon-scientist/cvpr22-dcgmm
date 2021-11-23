"""
"""
import math
import numpy                  as np
import tensorflow             as tf

from layer                      import GMM_Layer
from utils                      import log

from .regularizer.SingleExp     import SingleExp
from .regularizer.Time_Decay    import TimeDecayReg
from .regularizer.Regularizer   import Regularizer_Method as RM


class Energy:
  LOGLIK     = 'loglik'
  MC         = 'mc'


class MFA_Layer (GMM_Layer):
  """
  @param prev:    previous layer instance NOT a tensor
                  (shape: [batch_size=?, h_out, w_out, c_out], e.g., [?, 1, 1, 784] for MNIST image without folding)
  @param kwargs:  parameter dictionary with all parameters (all layers),
                  layer specific parameters are in self.layer_params (extracted in base class)
  """
  def __init__(self, input=None, **kwargs):

    GMM_Layer.__init__(self, input=input, **kwargs)
    parser = self.parser ;

    self.l                     = parser.add_argument('--l'                , type=int  , default=-1                       , help='latent space dim for MFA')
    self.latent_stddev         = parser.add_argument('--latent_stddev'    , type=float, default=1.                       , help='stddev in latent space for MFA')
    self.lambda_E              = parser.add_argument('--lambda_E'         , type=float, default=1.                       , help='factor for the Ds in MFA')
    self.lambda_gamma          = parser.add_argument('--lambda_gamma'     , type=float, default=1.                       , help='factor for the Ds in MFA')
    self.init_factor           = parser.add_argument('--init_factor'      , type=float, default=0.1                       , help='initial values of eigenvalues of M are 1-initFactor')
    self.mfa_mode              = parser.add_argument('--covariance_mode'  , type=str  , choices=["precision", "variance"], default='precision'               , help='precision or variance in MFA')



  def get_layer_variables(self, all_=False):
    ''' This saves all trainable parameters of the current layer into a dictionary to return '''
    variables = {        # all trainable variables
      f'{self.prefix}mus'   : self.mus   ,
      f'{self.prefix}gammas': self.gammas,
      f'{self.prefix}pis'   : self.pis   ,
      f'{self.prefix}Es'    : self.E   ,
      }
    if all_:
      variables.update({ # add other variables to load/create checkpoints
      f'{self.prefix}somSigma': self.tfSomSigma,
      f'{self.prefix}tfEps'   : self.tfEps     ,
        })
    return variables


  '''
  randomly initialize  gammas for a single colponent.
  -gamma_i gamma_i must be small
  -gammai gamma_j = 0 (orthogonal)
  '''
  def __orthoColMat(self,factor): # returns a matrix with c_in rows and self.l cols. cols are orthogonal and$

    # create matrix withn random orthogonal columns
    # random
    initUnit    = np.random.uniform(-1, 1, size=[self.c_in, self.l]).astype(self.dtype_np_float)
    # orthogonalize
    initUnit, _ = np.linalg.qr(initUnit)

    # normalize cols to length 1
    initUnit /= (initUnit*initUnit).sum(axis=0,keepdims=True)

    # factor should be small so that diagonla elements of M are small
    ret         = factor * initUnit

    return ret

  # TODO switch for var/prec
  def compute_M(self):
    self.AT         = tf.transpose(self.gammas, perm=(0, 1, 2, 3, 5, 4))                                    # 5,784
    self.A          = self.gammas                                                                           # 784,5
    coreM = tf.matmul(self.AT, self.A / (self.E[..., tf.newaxis]))
    M     = 1* self.I - coreM                                                                    # 1,pY,pX,K,l,D * 1,pY,pX,K,D,l --> 1,pY,PX,K,l,l
    return M ;



  def compile(self):
    GMM_Layer.compile(self) ;
    self.lambda_E_factor      = self.variable(self.lambda_E, shape=[],  name='lambda_E')
    self.lambda_gamma_factor  = self.variable(self.lambda_gamma, shape=[],  name='lambda_gamma')
    self.lambda_mu_factor     = self.variable(self.lambda_mu, shape=[],  name='lambda_mu')
    self.lambda_pi_factor     = self.variable(self.lambda_pi, shape=[],  name='lambda_pi')

    # define shapes
    pis_shape                = [1, 1, 1, self.K]
    mus_shape                = [1, 1, 1, self.K, self.c_in]
    # loading matrices
    gammas_shape             = [1, 1, 1, self.K, self.c_in, self.l]
    # diagonal cov matrix part
    E_shape                  = [1, 1, 1, self.K, self.c_in]

    # var initializers
    init_pi           = self.constant_initializer(1. / self.K)
    init_eps          = self.constant_initializer(self.eps0)
    init_E            = self.constant_initializer(math.sqrt(self.sigmaUpperBound)) ;
    init_somSigma     = self.constant_initializer(self.somSigma0)
    init_rand_mu      = tf.initializers.RandomUniform(-self.muInit, +self.muInit)
    #init_gammas       = self.constant_initializer(math.sqrt(self.sigmaUpperBound)
    initFactor        = math.sqrt(self.init_factor) ;
    # array of K conjugate loading matrices
    init_gammas       = np.array([self.__orthoColMat(initFactor) for k in range(self.K)]).reshape(gammas_shape) ;


    self.tfEps               = self.variable(initial_value=init_eps(shape=[]),            shape=[],        name='eps')
    self.tfSomSigma          = self.variable(initial_value=init_somSigma(shape=[]),       shape=[],        name='somSigma')
    # the raw pis, before used they are passed through a softmax
    self.pis                 = self.variable(initial_value=init_pi(shape=pis_shape),      shape=pis_shape, name='pis')
    self.mus                 = self.variable(initial_value=init_rand_mu(shape=mus_shape), shape=mus_shape, name='mus')
    self.E                   = self.variable(initial_value=init_E(shape=E_shape), shape=E_shape, name='E')
    # TODO initialize sigmas for MFA!!
    # sigmas are conjugate loading matrices
    # initialize their columns to be orthogonal and D to be 
    self.gammas           = self.variable(initial_value=init_gammas, name='gammas')
    # unit matrix for convenience
    self.I                = tf.cast(tf.linalg.diag(tf.ones(shape=(1, 1, 1, self.K, self.l))), self.dtype_tf_float)
    #self.M                = self.compute_M()
    log.debug(f"TEST!!!{self.init_factor}")



  # raw input --> log-scores, i.e. log (p_k p_k) --> N,pY,pX,K
  @tf.function(autograph=False)
  def forward(self, input_tensor, extra_inputs = None):   # input_tensor is the mini-batch, runs eagerly
      #tf.print(f'input tensor shape={input_tensor.shape}')
      const_    = -self.c_in / 2. * tf.math.log(2. * math.pi)                                                           # --> N,pY,pX,K
      diffs     = tf.expand_dims(input_tensor, 3) - self.mus                                                            # (N, pY,pX, 1, D)  - (1,pY,pX,K,D) --> (N,pY,pX,K,D)
      self.M    = self.compute_M()

      if self.mfa_mode == 'variance':
        # code der mit Varianzen und Kovarianzen arbeitet. Matmul ersetzt durch eigne Routine aufgrund von cuda10.0 bug in cublas. formeln aus richardson 2018 oder mclachlan 2003
        # ---------- diffs is (x-mu)^T
        logdet = -0.5 * tf.math.log((tf.linalg.det(self.M)) - tf.reduce_sum(tf.math.log(self.E), axis=4)) + const_        # K
        x0     = tf.matmul(AT, (diffs / (self.E))[..., tf.newaxis])                                          # K,l,D *  (K,D,1)  --> K,l,1

        x2     = tf.matmul((tf.linalg.inv(self.M)), x0)                                                 # K, l,l * K,l --> K,l,1
        x3     = tf.matmul(self.A, x2)                                                                       # d,1
        x4     = x3 / (self.E[..., tf.newaxis])
        x5     = tf.reduce_sum(diffs[..., tf.newaxis] * (x4), axis=(4, 5))
        logexp = + 0.5 * x5 - 0.5 * tf.reduce_sum((diffs ** 2.) / (self.E), axis=4)                

      if self.mfa_mode == 'precision':
        # ------------------------------------
        # Code der mit precision, atrices arbeitet. Numerisch unproblematisch ausser dass L pos. def. bleiben muss das muss explizit geprueft und durchgesetzt werden
        logdet     = 0.5 * tf.math.log(tf.linalg.det(self.M)) + 0.5 * tf.reduce_sum(tf.math.log(self.E), axis=4)     # K
        x0         = tf.matmul(self.AT, diffs[..., tf.newaxis])                                              # K,l no matmul, WORKS!!
        x1         = tf.reduce_sum(x0 ** 2, axis=(4, 5))                                                   # K
        logexp     = 0.5 * x1 - 0.5 * tf.reduce_sum((diffs ** 2.) * (self.E), axis=4)
      # ---------------------------------------    

      log_probs = (logdet + logexp)                                                                                   # --> N,pY,pX,K
      exp_pis   = tf.exp(self.pis)                                                                                      # --> 1,1,1,K # obtain real pi values by softmax over the raw pis thus, the real pis are always positive and normalized
      real_pis  = exp_pis / tf.reduce_sum(exp_pis)

      log_scores = tf.math.log(real_pis) + log_probs                               # --> N,pY,pX,K

      return log_scores

  # and returns new M
  def apply_constraints(self, M):

    if self.mfa_mode == 'precision':
      # sigma clipping for diag matrix E!
      E_limit = math.sqrt(self.sigmaUpperBound)
      self.E.assign(tf.clip_by_value(self.E, -E_limit, E_limit))

      M = self.compute_M()
  
      # gamma/M "clipping"
      # zur umkehrung der Gradientenrichtung wenn L in Gefahr laeuft neg def zu werden
      diag, O = tf.linalg.eigh(M)

      # diagonalize M by multiplyin gamma by 0
      #OT              = tf.transpose(O, (0, 1, 2, 3, 5, 4))
      newGammas       = tf.matmul(self.gammas, 1. * O)
      self.gammas.assign(newGammas)

      # eigenvals of M are unaffected by the last op, so use them further
      # if an eigenval is smaller than thr2 --> modify gammas!
      
      thr2       = 0.05

      # for every component: which diagonla elements of M are smaller than thr2? --> 1,1,1,K,l
      indicator  = tf.cast(tf.greater(tf.cast(thr2, self.dtype_tf_float), diag), self.dtype_tf_float)
      expanded_indicator = tf.expand_dims(indicator, axis=4) ;  # 1,1,1,K,1,l
      # mult mask that leaves columns unchanged where eigenvals are ok, and lultiplies others by 0.9 so that eigenvals get larger
      mask = (1. - expanded_indicator) + 0.9 * expanded_indicator
      self.gammas.assign(self.gammas * mask)
      # gradient retraction constraint, obsolete?
      #ind2  = tf.cast(tf.less(indicator, tf.cast(0.5, self.dtype_tf_float)), self.dtype_tf_float) * 2 - 1
      #self.mask  = ind2[..., tf.newaxis, tf.newaxis]

    elif self.mfa_mode == 'variance':
      raise Exception("variance only partially implemented") ;

  def pre_train_step(self):
    if self.percentage_task_done < self.wait: 
      self.lambda_pi_factor.assign( 0.0) ;
      self.lambda_E_factor.assign( 0.0) ;
      self.lambda_gamma_factor.assign( 0.0) ;
      self.lambda_mu_factor.assign(0.0) ;
    else:
      self.lambda_pi_factor.assign( self.lambda_pi) ;
      self.lambda_E_factor.assign( self.lambda_E) ;
      self.lambda_gamma_factor.assign( self.lambda_gamma) ;
      self.lambda_mu_factor.assign(self.lambda_mu) ;


  def update_with_grad(self, grads):

    self.pis.assign_add(self.lambda_pi_factor * self.tfEps * grads[self.prefix+'pis'])
    self.mus.assign_add(self.lambda_mu_factor * self.tfEps * grads[self.prefix+'mus'])
    self.gammas.assign_add(self.lambda_gamma_factor * self.tfEps * grads[self.prefix+'gammas'])
    self.E.assign_add(self.lambda_E_factor * self.tfEps * grads[self.prefix+'Es'])


  def post_train_step(self):
    # ---------------- CONSTRAINT ENFORCEMENT
    self.apply_constraints(self.M) ;

    # ---------------- regularizer update
    if self.percentage_task_done >= self.wait:
      last_log_likelihood = tf.reduce_mean(self.raw_loss) # batch/patch mean o$
      self.reg.add(last_log_likelihood).check_limit()
  
  # TODO: Fix this, does not yet work!!!
  ''' 
  Backward pass: sampling operator (create a batch of samples from the prototypes)
    - lower layer: N, h, w, cIn
    - upperLayer : N, h, w, cOut
    - input      : fromUpperLayer = N, h, w, cOut
    - output     : N, h, w, cIn
  GMM components are picked over probabilities pi (e.g. pick 3 components: [0.8, 0.1, 0.1]
  First component is contained in 80% of all samples, second in 10%, etc...
  This procedure is hierarchical, samples are constructed from typical outputs of the underlying layer, 
  Outputs gets passed until back to first layer of the model.
  '''
  def backwards(self, topdown, *args, **kwargs):

    # -------
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

    # -------------------
    # select mus and sigmas according to the post-processed topdown tensor need to distinguish between convMode (only one set of mus/sigmas for all RFs) and ind mode (separate mus/sigmas forall RFs)
    selectedMeansTensor  = tf.gather(self.mus[0, 0, 0]   , selectorsTensor, axis=0, batch_dims=0)   # select only the prototypes --> N , D
    selectedGammasTensor = tf.gather(self.gammas[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)   # --> N, ?
    selectedEsTensor = tf.gather(self.E[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)   # --> N, ?
    selectedMsTensor = tf.gather(self.compute_M()[0, 0, 0], selectorsTensor, axis=0, batch_dims=0)   # --> N, ?
    

    print ("6666666666666666666666666666666666666666666666666")

    if self.mfa_mode == 'precision':
      latentVars           = tf.random.normal(
        shape  = [N, self.l, 1]     ,
        mean   = 0.                 ,
        stddev = 1./self.sampling_divisor ,
        dtype  = self.dtype_tf_float,
        )
      selectedMeansTensor  = tf.reshape(selectedMeansTensor, (N, cIn, 1))
      selectedGammasTensor = tf.reshape(selectedGammasTensor,(N, cIn, self.l))
      selectedMsTensor     = tf.reshape(selectedMsTensor, (N,self.l, self.l))
      selectedETensor      = tf.reshape(selectedEsTensor, (N,self.c_in,1))
      vals, O              = tf.linalg.eigh(selectedMsTensor)
      OT                   = tf.transpose(O, (0, 2, 1))                                           # TODO: never used
      # use freedom in choice of sigmas to assume that L is diagonal
      sqrtM                = tf.linalg.diag(tf.sqrt(1. / vals))
      trafo                = tf.matmul(selectedGammasTensor / selectedETensor, sqrtM)          # TODO: never used
      upgrade              = selectedMeansTensor + tf.matmul(trafo, latentVars)
      mvnTensor            = tf.cast(upgrade, self.dtype_tf_float)                                # TODO: never used
    if self.mfa_mode == 'variance':
      latentVars           = tf.random.normal(
        shape  = [N * self.l * 1]   ,
        mean   = 0.                 ,
        stddev = self.latent_stddev ,
        dtype  = self.dtype_tf_float,
        )
      latentVars           = tf.reshape(latentVars, (N, self.l, 1))
      selectedMeansTensor  = tf.reshape(selectedMeansTensor, (N, cIn, 1))
      selectedSigmasTensor = tf.reshape(selectedSigmasTensor, (N, cIn, self.l))
      upgrade              = selectedMeansTensor + tf.matmul(selectedSigmasTensor, latentVars)
      mvnTensor            = tf.cast(upgrade, self.dtype_tf_float)


    sampling_op = tf.reshape(upgrade, (N, h, w, cIn))
    return sampling_op*sampling_mask


