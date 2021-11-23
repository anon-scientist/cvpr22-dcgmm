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
import os
import math
import gzip
import pickle
from scipy      import ndimage
import numpy    as np
from tensorflow import data
from DCGMM.parsers    import Kwarg_Parser
from DCGMM.utils      import log
import tensorflow_datasets as tfds 

Dataset_Type = [
  'SVHN'        ,
  'MNIST'       ,
  'CUB200'      ,
  'EMNIST'      ,
  'Fruits'      ,
  'CIFAR10'     ,
  'MADBase'     ,
  'NotMNIST'    ,
  'Devanagari'  ,
  'FashionMNIST',
  'ISOLET'      ,
  'CelebA'      ,
  ]


class TF2_Dataset(object):

  def __init__(self, **kwargs):
    self.parser                          = Kwarg_Parser(**kwargs)
    self.dataset_name                    = self.parser.add_argument('--dataset_name'                   , type=str  , default='mnist'                          , help='tfds download string')
    self.dataset_dir                     = self.parser.add_argument('--dataset_dir'                    , type=str  , default='./datasets'                     , help='set the default directory to search for dataset files or create them')
    self.dataset_file                    = self.parser.add_argument('--dataset_file'                   , type=str  , default='MNIST'  , help='load a compressed pickle file. If not present, a download attempt is made. This may take a while for large datasets such as SVHN')
    self.renormalize01                   = self.parser.add_argument('--renormalize01'                  , type=str  , default='no'     , choices=['no', 'yes'] , help='renormalize the data in a range [-1, +1] instead the "normal" range of [0, +1]')
    self.renormalizeC                    = self.parser.add_argument('--renormalizeC'                   , type=float  , default=255.                           , help='renormalize the data by dividing all channels by x')

    self.slice                           = self.parser.add_argument('--slice'                          , type=int  , default=[-1, -1]                         , help='replace all images in dataset by a N x M-patch cropped from the center of each image. if non negative, the two value represent N and M, otherwise the full image is used')
    self.squeeze                         = self.parser.add_argument('--squeeze'                        , type=int  , default=[-1, -1]                         , help='squeeze all images in dataset to a N x M-patch')
    self.decimate_class_train            = self.parser.add_argument('--decimate_class_train'           , type=int  , default=[]                               , help='defines the classes (train dataset) whose number of examples should be reduced')
    self.decimate_percentage_train       = self.parser.add_argument('--decimate_percentage_train'      , type=float, default=[]                               , help='define the reduction factor for train dataset (range [0.0, 1.0]), if only one reduction value is given, it is used for all defined (decimate_class_train) classes')
    self.decimate_class_test             = self.parser.add_argument('--decimate_class_test'            , type=int  , default=[]                               , help='defines the classes (test dataset) whose number of examples should be reduced')
    self.decimate_percentage_test        = self.parser.add_argument('--decimate_percentage_test'       , type=float, default=[]                               , help='define the reduction factor for test dataset (range [0.0, 1.0]), if only one reduction value is given, it is used for all defined (decimate_class_test) classes')
    self.noise_train                     = self.parser.add_argument('--noise_train'                    , type=float, default=0.                               , help='use noise factor to add noise to the complete training data (min, max) * factor')
    self.noise_test                      = self.parser.add_argument('--noise_test'                     , type=float, default=0.                               , help='use noise factor to add noise to the complete test data (min, max) * factor')

    self.data_type                       = self.parser.add_argument('--data_type '                     , type=int  , default=32, choices=[32,64]              , help='training batch size')
    self.batch_size                      = self.parser.add_argument('--batch_size'                     , type=int  , default=100                              , help='training batch size')
    self.test_batch_size                 = self.parser.add_argument('--test_batch_size'                , type=int  , default=self.batch_size                  , help='test batch size')

    self.rotation_per_iteration          = self.parser.add_argument('--rotation_per_iteration'         , type=float, default=0                                , help='rotate the input data (images) of 1 degree per training iteration')
    self.brightness_change_per_iteration = self.parser.add_argument('--brightness_change_per_iteration', type=float, default=0                                , help='increase or decrease the brightness of the input data (images) by 0.001 each iteration (max decrease/increase= -0.5/+0.5, data clipping [-1, 1])')

    _5_work_days = (    # tuple(amplitude, shift, noise) # @7000 iteration
      (.4, .6, 0.026),  # class 0
      (.3, .4, 0.026),  # class 1
      (.3, .3, 0.026),) # class 2
    self.distribution_change_per_task    = self.parser.add_argument('--distribution_change_per_task', type=eval, default=0, nargs='0 is off, else see 5 day example')
    # self.parser.add_argument('--TX_distribution'           , type=float, default=[1., 1., 1., 0, 0, 0, 0, 0, 0, 0], nargs='*')


    self.dataset_file += '' if self.dataset_file.endswith('.pkl.gz') else '.pkl.gz'
    self.dataset_path  = self.dataset_dir

    file_path          = os.path.join(self.dataset_path, self.dataset_file)

    """
    with gzip.open(file_path) as f:
      data = pickle.load(f)
    self.properties                = data['properties']
    self.raw_train_samples         = data['data_train']
    self.raw_train_labels          = data['labels_train']
    self.raw_test_samples          = data['data_test']
    self.raw_test_labels           = data['labels_test']
    print (self.properties)
    """

    print ("Loading ", self.dataset_name) ;
    (xtr,ytr),info_tr = tfds.load(self.dataset_name, batch_size=-1, split="train",as_supervised=True, with_info=True)
    (xtst,ytst),info_test= tfds.load(self.dataset_name, batch_size=-1, split="test",as_supervised=True, with_info=True)
    h,w,c = info_test.features['image'].shape 
    num_classes = info_test.features['label'].num_classes
    print ("CCCCC=",info_test.features) ;

    if self.data_type==32:
      dt = np.float32
    else:
      dt = np.float64

    ytr_np = ytr.numpy().astype("int64") ;
    ytst_np = ytst.numpy().astype("int64") ;
    onehot_tr_raw = np.zeros([ytr_np.shape[0],num_classes],dtype=dt)
    onehot_tst_raw = np.zeros([ytst_np.shape[0],num_classes],dtype=dt)
    onehot_tr_raw[range(0,ytr_np.shape[0]),ytr_np] = 1 ;
    onehot_tst_raw[range(0,ytst_np.shape[0]),ytst_np] = 1 ;
    properties = {'num_of_channels':c, 'num_classes':10, 'dimensions':[h,w]} 
    self.properties                = properties ;
    self.raw_train_samples         = xtr.numpy().astype(dt).reshape(-1,h,w,c) ;
    self.raw_train_labels          = onehot_tr_raw
    self.raw_test_samples          = xtst.numpy().astype(dt).reshape(-1,h,w,c) ;
    self.raw_test_labels           = onehot_tst_raw ;

    print('raw train sample values are between [{}, {}]'.format(np.min(self.raw_train_samples), np.max(self.raw_train_samples)))
    print('raw test sample values are between [{}, {}]'.format(np.min(self.raw_test_samples), np.max(self.raw_test_samples)))
    

    # FIXME: re-normalize the data
    if self.renormalize01 == 'yes':
      lower, upper = 0, +1
      self.raw_train_samples = (upper - lower) * np.divide(
        np.subtract(self.raw_train_samples, np.min(self.raw_train_samples)),
        np.subtract(np.max(self.raw_train_samples), np.min(self.raw_train_samples))
      ) + lower
      self.raw_test_samples = (upper - lower) * np.divide(
        np.subtract(self.raw_test_samples, np.min(self.raw_test_samples)),
        np.subtract(np.max(self.raw_test_samples), np.min(self.raw_test_samples))
      ) + lower

    self.raw_train_samples /= self.renormalizeC ;
    self.raw_test_samples /= self.renormalizeC ;

    print('train sample values are between [{}, {}]'.format(np.min(self.raw_train_samples), np.max(self.raw_train_samples)))
    print('test sample values are between [{}, {}]'.format(np.min(self.raw_test_samples), np.max(self.raw_test_samples)))

    self.properties['train_shape'] = self.raw_train_samples.shape
    self.properties['test_shape']  = self.raw_test_samples.shape
    self.scalar_labels_train       = self.raw_train_labels.argmax(axis=1)
    self.scalar_labels_test        = self.raw_test_labels.argmax(axis=1)
    self.indices_train             = np.arange(self.raw_train_samples.shape[0])
    self.indices_test              = np.arange(self.raw_test_samples.shape[0])


  def get_iterator(self, type='training', enum=False, **kwargs):
    batch_size = kwargs.get('batch_size', 100)
    classes    = kwargs.get('classes', range(10))
    epochs     = kwargs.get('epochs', 1)
    ds_obj_train, ds_obj_test, _, _ = self.get_dataset(classes=classes, batch_size=batch_size, epochs=epochs)

    if type == 'training': return enumerate(iter(ds_obj_train)) if enum else iter(ds_obj_train)
    if type == 'testing' : return enumerate(iter(ds_obj_test))  if enum else iter(ds_obj_test) # always 1 epochs
    raise Exception('invalid type (default=training or testing)')


  def get_class_indices(self, classes):
    int_class  = int(classes)
    mask_train = (self.scalar_labels_train == int_class)
    mask_test  = (self.scalar_labels_test  == int_class)
    return self.indices_train[mask_train], self.indices_test[mask_test]


  def get_dataset(self, classes, **kwargs):
    ''' Returns TF dataset objects for train and test sets '''
    epochs     = kwargs.get('epochs', None) # infinity
    batch_size = kwargs.get('batch_size', None)
    test_batch_size = kwargs.get('test_batch_size', None)

    indices_set_train = []
    indices_set_test  = []
    for class_ in classes:
      indices_train, indices_test = self.get_class_indices(class_)
      indices_set_train += [indices_train]
      indices_set_test  += [indices_test]
    all_indices_train = np.concatenate(indices_set_train, axis=0)
    all_indices_test  = np.concatenate(indices_set_test, axis=0)

    np.random.shuffle(all_indices_train)
    np.random.shuffle(all_indices_test)

    data_train = self.raw_train_samples[all_indices_train]
    data_test  = self.raw_test_samples[all_indices_test]

    labels_train = self.raw_train_labels[all_indices_train]
    labels_test  = self.raw_test_labels[all_indices_test]

    h, w = self.properties['dimensions']
    c    = self.properties['num_of_channels']

    data_train_reshaped = data_train.reshape(-1, h, w, c)
    data_test_reshaped  = data_test.reshape(-1, h, w, c)

    # Construct a Dataset object (TF2) for drawing batches/shuffling here
    ds_obj_train = data.Dataset.from_tensor_slices((data_train_reshaped, labels_train))
    ds_obj_train = ds_obj_train.batch(batch_size if batch_size else self.batch_size , drop_remainder=True)
    ds_obj_train = ds_obj_train.repeat(epochs) # infinity if None (default)

    ds_obj_test  = data.Dataset.from_tensor_slices((data_test_reshaped, labels_test))
    ds_obj_test  = ds_obj_test.batch(test_batch_size if test_batch_size else self.test_batch_size, drop_remainder=True)

    return ds_obj_train, ds_obj_test, data_train.shape[0], data_test.shape[0]


  def rotate(self, xs, iteration, task, **kwargs):
    ''' rotate all images by a given angle (cmd param or kwarg) and iteration
    @param xs       : data
    @param iteration: iteration for task rotation (iteration % 360)
    @param task     : current training task
    '''
    task_rotation          = kwargs.get(f'T{task + 1}_rotation')
    if not (self.rotation_per_iteration or task_rotation): return xs # no rotation

    if self.rotation_per_iteration and self.rotation_per_iteratio != 0:
      task_rotation  = self.rotation_per_iteration
      rotation_angle = task_rotation * (iteration) % 360

    if task_rotation:
      rotation_angle = task_rotation

    if rotation_angle == 0.0: return xs

    return ndimage.rotate(xs, rotation_angle, reshape=False, axes=(2, 1))


  def brighten_darken(self, xs, iteration, task, **kwargs):
    ''' increase/decrease the brightness of images by a given value [-1., 1.] (clipping of images to [0., 1.]) '''
    task_brighten = kwargs.get(f'T{task + 1}_brightness')
    if not (self.brightness_change_per_iteration or task_brighten): return xs # no brighten

    if self.brightness_change_per_iteration != 0.:
      task_brightness = self.brightness_change_function()

    if task_brighten:
      task_brightness = task_brighten

    if task_brightness == 0: return xs
    return np.clip(xs + task_brightness, 0.0, 1.0)


  def brightness_change_function(self, toggle=+1.0, task_brightness=0., lower_limit=-0.5, upper_limit=0.5):
    ''' like scanner running light '''
    me               = self.brightness_change_function.__func__
    new_value        = task_brightness + (self.brightness_change_per_iteration * toggle)
    if new_value >= upper_limit or new_value <= lower_limit: me.__defaults__ = (me.__defaults__[0] * -1,) + me.__defaults__[1:]          # update toggle parameter
    task_brightness += self.brightness_change_per_iteration * toggle
    me.__defaults__  = (me.__defaults__[0], task_brightness) + me.__defaults__[2:] # update task_brightness parameter
    return task_brightness


  def distribution_change_function(self, task=0, tasks_per_day=10):
    ''' sinusoidal change of the class distribution (similar to network data distribution)

      @param tasks_per_day: number of tasks represent one day
      @return             : normed reduction parameter as list
    '''
    me              = self.distribution_change_function.__func__
    omega           = 1 / (tasks_per_day / (np.pi * 2))
    distribution    = list()

    for amplitude, shift, noise in self.distribution_change_per_task:
      value         = amplitude * np.cos(task * omega) + shift
      value        += np.random.normal(0, noise)   # add noise
      value         = np.clip(value, 0.0, 1.0)
      distribution += [value]
    me.__defaults__ = (me.__defaults__[0] + 1,) + me.__defaults__[1:]
    return distribution


  def change_distribution(self, dataset, iterations, current_task, **kwargs):
    ''' change the current distribution of the dataset
    @param dataset   : prepared dataset
    @param iterations: training iteration counter (1 epoch)
    @param task      : current training task
    '''
    task_distribution = kwargs.get(f'T{current_task + 1}_distribution')
    batch_size        = kwargs.get('batch_size', 100)
    if not (task_distribution or self.distribution_change_per_task): return dataset, iterations

    if task_distribution and np.any(np.array(task_distribution) > 1.): raise NotImplementedError('only a reduction (x < 1.) is possible')
    if task_distribution and np.any(np.array(task_distribution) < 0.): raise NotImplementedError('only a reduction (x >= 0.) is possible')

    if self.distribution_change_per_task:
      task_distribution = self.distribution_change_function(current_task)

    log.info(f'change dataset distribution for task {current_task} to {task_distribution}')
    data_iter = iter(dataset)
    _data = np.concatenate([next(data_iter)[0] for _ in range(iterations)], axis=0)

    data_iter = iter(dataset)
    labels = np.concatenate([next(data_iter)[1] for _ in range(iterations)], axis=0)

    reduced_labels       = list()
    reduced_data         = list()
    elements_all_classes = 0
    num_classes          = self.properties.get('num_classes', 10)
    task_distribution    += [ 1.0 for _ in range(num_classes - len(task_distribution)) ] # fill up not defined classes
    for class_, reduction in enumerate(task_distribution):
      mask                  = labels.argmax(axis=1) == class_
      num_elements_before   = np.sum(mask)
      num_elements_after    = int(num_elements_before * reduction)
      labels_               = labels[mask]
      reduced_labels       += [labels_[:num_elements_after]]
      data_                 = _data[mask]
      reduced_data         += [data_[: num_elements_after]]
      elements_all_classes += num_elements_after

    reduced_labels          = np.concatenate(reduced_labels)
    reduced_data            = np.concatenate(reduced_data)

    new_dataset = data.Dataset.from_tensor_slices((reduced_data, reduced_labels))
    new_dataset = new_dataset.batch(batch_size if batch_size else self.batch_size , drop_remainder=True)
    new_dataset = new_dataset.repeat()
    training_iter           = math.ceil(elements_all_classes / self.batch_size)
    return new_dataset, training_iter



