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
"""
visualizes data saved by GMM Layers --> GMM_Layer.py
Can vis, per component, depending on parameter "--what":
- centroids (mus)
- diagonal sigmas (sigmas)
- loading matrix rows (oadingMAtrix)
- convGMM centroids arranged in  visually intuitive way (organizedWeights)
Over all visualizations the pi value of that comp. can be overlaid.
"""
import numpy as np
import matplotlib as mp
from matplotlib import cm
import sys, gzip, pickle ;

# so as to not try to invoke X on systems without it
mp.use("Agg")

import matplotlib.pyplot as plt, sys, math
from argparse import ArgumentParser

def getIndexArray(w_in, h_in, c_in, w_out, h_out, c_out, patch_width, patch_height):
    indicesOneSample = np.zeros([w_in*h_in*c_in],dtype=np.int32) ;
    tileX = int(math.sqrt(c_in)) ; tileY = tileX ;
    tilesX = w_in ; tilesY = c_in ;
    sideX = tileX * w_in ; sideY = tileY * h_in ;

    for inIndex in range(0,w_in * h_in * c_in):
      tileIndexFlat = inIndex // c_in ;
      tileIndexX = tileIndexFlat % tilesX ;
      tileIndexY = tileIndexFlat // tilesX ;

      tilePosFlat = inIndex % (tileX*tileY) ;
      tilePosY = tilePosFlat // tileX ;
      tilePosX = tilePosFlat % tileX ;

      posY = tileIndexY * tileY + tilePosY ;
      posX = tileIndexX * tileX + tilePosX ;
      outIndex = sideX  * posY + posX ;

      indicesOneSample [outIndex] = inIndex ;
    return indicesOneSample ;


if __name__ == "__main__":
  plt.rc('text', usetex=False)
  plt.rc('font', family='serif')

  parser = ArgumentParser()
  parser.add_argument("--channels", required=False, default=1, type=int, help = "If u r visualizing centroids that come from color images (SVHN, Fruits), please specify 3 here!")
  parser.add_argument("--y", required=False, default=0, type=int, help = "PatchY index for convGMMs")
  parser.add_argument("--x", required=False, default=0, type=int, help = "PatchX index for convGMMs")
  parser.add_argument("--l", required=False, default=0, type=int, help = "row of MFA loading matrix")
  parser.add_argument("--what", required=False, default="mus", choices=["mus2D", "mus","precs_diag","organizedWeights","loadingMatrix"], type=str, help="Visualize centroids or precisions°")
  parser.add_argument("--prefix", required=False, default="gmm_layer_", type=str, help="Prefix for file names")
  parser.add_argument("--vis_pis", required=False, default=False, type=eval, help="True or False depending on whether you want the weights drawn on each component")
  parser.add_argument("--cur_layer_hwc", required=False, default=[-1, -1, -1], type=int, nargs = 3, help = "PatchX index for convGMMs")
  parser.add_argument("--prev_layer_hwc", required=False, default=[-1, -1, -1], type=int, nargs = 3, help = "PatchX index for convGMMs")
  parser.add_argument("--filter_size", required=False, default=[-1, -1], type=int, nargs = 2, help = "PatchX index for convGMMs")
  parser.add_argument("--proto_size", required=False, default=[-1, -1], type=int, nargs = 2, help = "PatchX index for convGMMs")
  parser.add_argument("--clip_range", required=False, default=[-100., 100.], type=float, nargs = 2, help = "clip display to this range")
  parser.add_argument("--disp_range", required=False, default=[-100., 100.], type=float, nargs = 2, help = "clip display to this range")
  parser.add_argument("--out", required=False, default="mus.png", type=str, help="output file name")
  parser.add_argument("--dataset_file", required=False, default ="", type=str, help="data points to plot for 2D visualisation")
  parser.add_argument("--mu_file", required=False, default ="mus.npy", type=str, help="data points to plot for 2D visualisation")
  parser.add_argument("--sigma_file", required=False, default ="sigmas.npy", type=str, help="data points to plot for 2D visualisation")
  parser.add_argument("--pi_file", required=False, default ="pis.npy", type=str, help="data points to plot for 2D visualisation")
  FLAGS = parser.parse_args()

  pifile = FLAGS.pi_file ;
  mufile = FLAGS.mu_file ;
  sigmafile = FLAGS.sigma_file ;
  if FLAGS.what == "loadingMatrix":
    sigmafile = "gammas.npy"

  channels = FLAGS.channels
  it = FLAGS.prefix ;

  protos = np.load(it + mufile) ; print ("Raw protos have shape", protos.shape) ;

  pis = sigmas = None ;
  if len(protos.shape) == 4: ## sampling file, single npy with dims N,d*d
    _n,_h,_w,_c = protos.shape ;
    _d = _h*_w*_c ;
    pis = np.zeros([_n,_d])
    sigmas = np.zeros([_n,_d])
    protos = protos.reshape(_n,_d)
  else:
    pis = np.load(it + pifile)[0, FLAGS.y, FLAGS.x]
    sigmas = np.load(it + sigmafile)[0,FLAGS.y, FLAGS.x]
    protos = protos[0, FLAGS.y, FLAGS.x]

  n = int(math.sqrt(protos.shape[0]))
  imgW = int(math.sqrt(protos.shape[1] / channels)) ;
  imgH = int(math.sqrt(protos.shape[1] / channels)) ;
  d_ = int(math.sqrt(protos.shape[1] / channels)) ;
  print (FLAGS.proto_size) ;
  if FLAGS.proto_size[0] != -1: 
    imgH, imgW = FLAGS.proto_size ;
    print (imgH,imgW, "!!!")

  print ("ND=", n,d_)

  h_in, w_in, c_in = FLAGS.prev_layer_hwc ;
  h_out, w_out, c_out = FLAGS.cur_layer_hwc ;
  pH,pW = FLAGS.filter_size ;

  indices = None ;
  if h_in != -1:
    indices = getIndexArray(h_in,w_in,c_in, h_out, w_out, c_out, pH, pW) ;
    print (indices.shape, "INDICES") ;
    print (indices.min(), indices.max(), "INDICES") ;

  if FLAGS.what == "mus2D":
    f,ax = plt.subplots(1,1) ;
    data = None;
    if FLAGS.dataset_file != "": 
      with gzip.open(FLAGS.dataset_file) as f:
        data = pickle.load(f)["data_test"] ;
        print ("Loaded data: ", data.shape)
    if data is not None: ax.scatter(data[:,0], data[:,1]) ;
    ax.scatter(protos[:,0],protos[:,1]) ;
    plt.tight_layout(pad=1, h_pad=.0, w_pad=-10)
    plt.savefig(FLAGS.out)
    sys.exit(0) ;



  f = axes=None;
  #f, axes = plt.subplots(n, n, gridspec_kw={'wspace':0, 'hspace':0}) ;
  f, axes = plt.subplots(n, n) ;
  if n ==1:
    f = np.array([f]) ;
    axes = np.array([axes]) 

      


  
  axes = axes.ravel()
  index = -1

  exp_pi = np.exp(pis)
  sm = exp_pi/exp_pi.sum()
  for (dir_, ax_, pi_, sig_) in zip(protos, axes, sm, sigmas):
    index += 1

    disp = dir_
    if FLAGS.what == "precs_diag":
      disp = sig_
    if FLAGS.what == "loadingMatrix":
      disp = sig_[:,FLAGS.l]
      print (disp.shape);

    if FLAGS.what == "organizedWeights" and indices is not None:
      #dispExp = np.exp(disp.reshape(h_in, w_in, c_in)) ; print (dispExp.shape) ;
      #disp = dispExp / dispExp.sum(axis=2,keepdims=True) ;
      dispExp = (disp.reshape(h_in, w_in, c_in)) ; print (dispExp.shape) ;
      disp = dispExp ;
      disp = disp.ravel()[indices] ;
      print ("Disp hape=",disp.shape);

    disp = np.clip(disp,FLAGS.clip_range[0],FLAGS.clip_range[1]) ;

    refmin = disp.min() if FLAGS.disp_range[0] == -100 else FLAGS.disp_range[0];
    refmax = disp.max() if FLAGS.disp_range[1] == +100 else FLAGS.disp_range[1];


    # This is interesting to see unconverged components
    print(index, "minmax=", disp.min(), disp.max(), refmin,refmax, disp.shape, channels, imgH, imgW)

    ax_.imshow(disp.reshape(imgH, imgW, channels) if channels == 3 else disp.reshape(imgH,imgW), vmin=refmin, vmax=refmax, cmap=cm.bone)

    if FLAGS.vis_pis == True:
      ax_.text(-5, 1, "%.03f" % (pi_), fontsize=5, c="black", bbox=dict(boxstyle="round", fc=(1, 1, 1), ec=(.5, 0.5, 0.5)))

    ax_.set_aspect('auto')
    ax_.tick_params( # disable labels and ticks
      axis        = 'both',
      which       = 'both',
      bottom      = False ,
      top         = False ,
      left        = False ,
      right       = False ,
      labelbottom = False ,
      labelleft   = False ,
      )

  #plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0);
  plt.tight_layout(pad=0, h_pad=0.25, w_pad=0.25)
  #plt.tight_layout()
  plt.savefig(FLAGS.out)
  #plt.savefig("mus.pdf")

