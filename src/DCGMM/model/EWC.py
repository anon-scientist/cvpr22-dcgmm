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
from utils             import log
from model             import Model
from metric            import Metric
import tensorflow      as tf
import numpy           as np
from collections       import defaultdict

class EWC(Model):

  def __init__(self, **kwargs):
    super(EWC, self).__init__(**kwargs)

    self.eps           = self.parser.add_argument('--epsilon'      , type=float, default=0.00001            , help='learn rate')
    self.lambda_       = self.parser.add_argument('--lambda'       , type=float, default=1. / self.eps      , help='EWC lambda')
    self.hidden_layers = self.parser.add_argument('--hidden_layers', type=int  , default=2                  , help='number of hidden layers')
    self.layer_size    = self.parser.add_argument('--layer_size'   , type=int  , default=200                , help='size of each hidden layer')
    self.input_size    = self.parser.add_argument('--input_size'   , type=int  , default=784                , help='input dimenions')
    self.num_classes   = self.parser.add_argument('--num_classes'  , type=int  , default=10                 , help='input dimenions')

    # evaluation metrics
    metrics               = ['accuracy_score'] # + ['loss']
    self.metrics          = self.parser.add_argument('--metrics'      , type=str  , default=metrics               , help='the evaluations metrics')
    self.metric           = Metric(self)

    self.current_task = 0
    self.init_tf_variables()

    log.debug(f'lambda={self.lambda_}')


  def randomize_weights(self): # is called by before_task method
    for (W, b) in self.internal_layers:
      W.assign(tf.random.truncated_normal(W.shape, 0, 0.1, dtype=self.dtype_tf))
      b.assign(tf.random.truncated_normal(b.shape, 0, 0.1, dtype=self.dtype_tf))


  def init_tf_variables(self):
    self.internal_layers      = []
    self.ewc_storage = {}  # dict indexed by task (T0 - Tx-1), has elements of [W,b]
    self.fims        = {}  # dict indexed by task (T0 - Tx-1)
    self.var_list    = []
    self.fim_acc     = []

    prev_size        = self.input_size
    cur_size         = self.layer_size
    for l in range(self.hidden_layers): # TODO: convert to a Layer class
      W                     = tf.Variable(tf.random.truncated_normal(shape=[prev_size, cur_size], mean=0, stddev=0.1, dtype=self.dtype_tf_float))
      b                     = tf.Variable(tf.random.truncated_normal(shape=[1, cur_size]        , mean=0, stddev=0.1, dtype=self.dtype_tf_float))
      log.debug(f'allocating W for layer {l+1} w shape {W.shape}')

      accW                  = tf.Variable(tf.random.truncated_normal(shape=[prev_size, cur_size], mean=0, stddev=0.1, dtype=self.dtype_tf_float))
      accb                  = tf.Variable(tf.random.truncated_normal(shape=[1, cur_size]        , mean=0, stddev=0.1, dtype=self.dtype_tf_float))

      self.internal_layers += [[W   , b   ]]
      self.fim_acc         += [[accW, accb]]
      prev_size             = cur_size
      cur_size              = self.layer_size

    Wlr            = tf.Variable(tf.random.truncated_normal(shape=[self.layer_size, self.num_classes], mean=0, stddev=0.1, dtype=self.dtype_tf_float))
    blr            = tf.Variable(tf.random.truncated_normal(shape=[1, self.num_classes]              , mean=0, stddev=0.1, dtype=self.dtype_tf_float))
    log.debug(f'allocating W for layer {self.hidden_layers+1} w shape {Wlr.shape}')

    accWlr         = tf.Variable(tf.random.truncated_normal(shape=[self.layer_size, self.num_classes], mean=0, stddev=0.1, dtype=self.dtype_tf_float))
    accblr         = tf.Variable(tf.random.truncated_normal(shape=[1, self.num_classes]              , mean=0, stddev=0.1, dtype=self.dtype_tf_float))

    self.internal_layers   += [[Wlr   , blr   ]]
    self.fim_acc  += [[accWlr, accblr]]

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eps)


  def get_model_variables(self):
    ''' export for loading and saving '''
    variables = {}
    variables.update({f'W{i}': layer[0] for i, layer in enumerate(self.internal_layers)})
    variables.update({f'b{i}': layer[1] for i, layer in enumerate(self.internal_layers)})

    for task, fims in self.fims.items():
      variables.update({f'fim_task{task}_W{i}': fim[0] for i, fim in enumerate(fims)})
      variables.update({f'fim_task{task}_b{i}': fim[1] for i, fim in enumerate(fims)})

    for task, storage in self.ewc_storage.items():
      variables.update({f'stored_task{task}_W{i}': store[0] for i, store in enumerate(storage)})
      variables.update({f'stored_task{task}_b{i}': store[1] for i, store in enumerate(storage)})
    return variables


  def forward(self, xs):
    xs = tf.reshape(xs, (-1, self.input_size))
    for W, b in self.internal_layers[0: -1]:
      xs = tf.nn.relu(tf.matmul(xs, W) + b)
    Wlr, blr = self.internal_layers[-1]
    logits   = tf.matmul(xs, Wlr) + blr
    return logits


  def build(self, **kwargs):
    log.info('build EWC model')
    return self


  def loss(self, xs, ys):
    self.ewc_loss = 0.0
    for task in range(self.current_task):
      for (W, b), (W_prev, b_prev), (fim_W_prev, fim_b_prev) in zip(self.internal_layers, self.ewc_storage[task], self.fims[task]):
        self.ewc_loss = self.ewc_loss + tf.reduce_sum((W - W_prev) ** 2 * fim_W_prev)
        self.ewc_loss = self.ewc_loss + tf.reduce_sum((b - b_prev) ** 2 * fim_b_prev)
    self.dnn_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=xs, labels=ys))
    return self.dnn_loss + self.lambda_ * self.ewc_loss


  def compute_gradients(self, xs, ys, dnn_only=False):
    with tf.GradientTape() as g:
      logits = self.forward(xs)
      loss   = self.loss(logits, ys)
    if dnn_only: return g.gradient(self.dnn_loss, self.internal_layers)
    else       : return g.gradient(loss         , self.internal_layers)


  def test_step(self, xs, ys=None, **kwargs):
    collect_results = self.evaluate(xs, ys)

    layer_metrics = self.post_test_step(    # compute layer test steps
      results = collect_results.get('EWC'), # output of testing, e.g. log-likelihood, accuracy, outliers
      xs      = xs                        , # the batches used for evaluation as list
      ys      = ys                        ,
    )

    return {'EWC': layer_metrics}


  def test(self, test_iterator, **kwargs):
    ''' test the model with the given test iterator
    @param test_iterator: TF2 test iterator
    @return: dict(tuple(source <str>, metric <str>: tuple(metric_value_raw <float>, formatted metric value <str>)))
    '''
    results = defaultdict(list)
    for xs, ys in test_iterator:
      test_results = self.test_step(xs, ys)

      for layer_name, metric_and_values in test_results.items():
        for metric_name, metric_value in metric_and_values.items():
          results[(layer_name, metric_name)] += [metric_value]

    return_results = dict()
    for (layer_name, metric_name), metric_values in results.items():
      if 'accuracy' in metric_name: format_str = '{:10.1%}'
      else                        : format_str = '{:10.2f}'
      metric_values                             = np.mean(metric_values)
      return_results[(layer_name, metric_name)] = (metric_values, format_str.format(metric_values))
    return return_results



  def train_step(self, xs, ys, **kwargs):
    gradients        = self.compute_gradients(xs, ys)
    unfold_gradients = []
    for (grW, grb), (W, b) in zip(gradients, self.internal_layers): # unfold gradients and create a flat list of (grad,var) pairs
      unfold_gradients += [(grW, W), (grb, b)]
    self.optimizer.apply_gradients(unfold_gradients)


  def set_parameters(self, **kwargs):
    self.current_task = kwargs['current_task']


  def start_fim_calculation(self):
    log.debug(f'FM computation, current task={self.current_task}')
    for fim_W, fim_b in self.fim_acc:     # reset FIM accumulators
      fim_W.assign(fim_W * 0.0)
      fim_b.assign(fim_b * 0.0)


  def update_fim_with_minibatch(self, X, Y):
    ''' Computes Fim/MasQ for one batch of samples and adds results to accumulator '''
    gr = self.compute_gradients(X, Y, dnn_only=False)
    for (gW, gb), (accW, accb) in zip(gr, self.fim_acc):
      accW.assign_add(gW ** 2)
      accb.assign_add(gb ** 2)


  def finalize_fim_calculation(self, nr_batches):
    ''' Stores current Ws and bs under the index of the current task. Assumed to be called AFTER training on the current task and AFTER fim accumulation '''
    for l, (fim_W, fim_b) in enumerate(self.fim_acc): # divide all accumulators by the number of batches
      fim_W.assign(fim_W / nr_batches)
      log.debug(f'fimW{l} min,max={tf.reduce_min(fim_W)}, {tf.reduce_max(fim_W)} --> {nr_batches}')
      fim_b.assign(fim_b / nr_batches)

    # copy variables
    self.fims[self.current_task]        = [(tf.constant(fim_W + 0.), tf.constant(fim_b + 0.)) for fim_W, fim_b in self.fim_acc]
    self.ewc_storage[self.current_task] = [(tf.constant(W + 0.)    , tf.constant(b     + 0.)) for (W, b)       in self.internal_layers ]


  def post_test_step(self, results, xs, ys=None, **kwargs):
    loss          = results.get('loss')
    y_pred        = results.get('y_pred')
    y_true        = results.get('y_true')

    metric_values = self.metric.eval(
      dict    = True  , # return a dictionary with metric values
      loss    = loss  ,
      y_true  = y_true,
      y_pred  = y_pred,
      special = {
        'accuracy_score': dict(normalize=True),
        }
      )
    return metric_values


  def evaluate(self, xs, ys=None, **kwargs):
    logits   = self.forward(xs)
    loss     = self.loss(logits, ys)

    y_pred   = tf.argmax(logits, axis=1)
    y_true   = tf.argmax(ys, axis=1)

    accuracy = tf.equal(y_true, y_pred)
    accuracy = tf.cast(accuracy, self.dtype_tf_float)
    accuracy = tf.reduce_mean(accuracy)

    result   = {
      'loss'    : loss    ,
      #'accuracy': accuracy,
      'y_true'  : y_true  ,
      'y_pred'  : y_pred  ,
      }
    return {'EWC': result}


  def reset_model(self, task):
    self.current_task = task


  def apply_imm_after_task(self, mode, imm_transfer_type, imm_alpha, current_task):
    if mode == 'ewc'    : return
    if current_task == 0: return

    prev_task_weight = 1. - imm_alpha
    cur_task_weight  = imm_alpha
    if mode == 'mean_imm':
      for (W, b), (prev_W, prev_b) in zip(self.internal_layers, self.ewc_storage[current_task - 1]):
        W.assign(prev_task_weight * prev_W + cur_task_weight * W)
        b.assign(prev_task_weight * prev_b + cur_task_weight * b)

    if mode == 'mode_imm':
      for (W, b), (prev_W, prev_b), (oldfim_W, oldfim_b), (fim_W, fim_b) in zip(self.internal_layers, self.ewc_storage[current_task - 1], self.fims[current_task - 1], self.fims[current_task]):

        commonW = prev_task_weight * oldfim_W + cur_task_weight * fim_W + 1e-30
        commonb = prev_task_weight * oldfim_b + cur_task_weight * fim_b + 1e-30

        newW    = (prev_task_weight * prev_W * oldfim_W + cur_task_weight * W * fim_W) / commonW
        newb    = (prev_task_weight * prev_b * oldfim_b + cur_task_weight * b * fim_b) / commonb
        W.assign(newW)
        b.assign(newb)
