# plotting outliers detection, very specuific setting
# not a general solution!

"""
evaluates a JSONS file for outlier detection.
Evaluated are test data, as JSON log files mainly record test data.
We assume that two tasks are existing in JSON file, T1 and T2.
T1 is supposed to be inliers, T2 outliers.  
So if we want to train outluer detection on MNIST, T1 would be 1-9, T2 0.

Generates two files if specified:
out.png - a plot of score-vs-iteration
toc.png - a ROC-like plot of inliers_kep - vs- outliers_rejected
"""

import json, numpy as np, sys ;
import matplotlib.pyplot as plt ;
from argparse import ArgumentParser ;

class OutlierComputation(object):

  def __init__(self, **kwargs):
    self.make_plots = kwargs.get("make_plots") ;
    self.json = kwargs.get("json") ;
    self.valid_data = True ;

    self.data = json.load(open(self.json,"r")) ;
    #print (data)
    eval_keys = self.data["eval"].keys() ;

    try:
      dT1 = self.data["eval"]["scores_T1_outliers" ]
      dT2 = self.data["eval"]["scores_T2_outliers" ]
      self.scoresT1 =  (dT1[1][2])
      self.scoresT2 =  (dT2[1][2])
    except Exception:
      self.valid_data = False ; 

  def get_scores(self):
    return self.scoresT1, self.scoresT2 ;

  def create_plots(self):
    if self.valid_data == False: return ;
    scoresT1 = self.scoresT1 ; 
    scoresT2 = self.scoresT2 ; 

    if self.make_plots:
      f,ax = plt.subplots(1,1) ;

      ax.scatter(range(0, len(scoresT1)), scoresT1 ) ;
      ax.scatter(range(0, len(scoresT2)), scoresT2 ) ;
      f.tight_layout() ;
      plt.savefig("out.png") ;
      ax.cla() ;

    if self.make_plots:
      ax.scatter(self.x,self.y) ;
      ax.set_xlabel("fraction inliers kept", fontsize=20)
      ax.set_ylabel("fraction outliers rejected", fontsize=20)
      ax.tick_params(axis='both', labelsize=20) ;
      ax.set_ylim(0.0,1.0)
      ax.set_xlim(0.0,1.0)
      f.tight_layout() ;
      plt.savefig("toc.png")
    


  def compute_auc(self):
    if self.valid_data == False: return None ;
    scoresT1 = np.array(self.scoresT1) ;
    scoresT2 = np.array(self.scoresT2) ;
    min1 = scoresT1.min() ;
    max1 = scoresT1.max() ;

    min2 = scoresT2.min() ;
    max2 = scoresT2.max() ;

    mn = min(min1, min2)
    mx = max(max1,max2)

    steps = 100 ;
    self.x = np.zeros([steps]) ;
    self.y = np.zeros([steps]) ;
    for step in range(0,steps):
      thr = mn + step * (mx-mn)/(steps+0.) ;
  
      # T1 scores must be inliers, T2 outliers
      # count how many T1 are above thr
      T1above = (scoresT1>thr).astype("float32").mean() ;
      T2below = (scoresT2<thr).astype("float32").mean() ;
      self.x[step] = T1above ;
      self.y[step] = T2below ;

    # area under curve computation
    auc = 0.0 ;
    last_x = None; 
    last_y = None ;
    indices = np.argsort(self.x) ;
    for _x,_y in zip(self.x[indices],self.y[indices]):
      if last_x is not None:
        auc += (_x-last_x) * (last_y)  + 0.5 * (_x-last_x) * (_y-last_y) ;
        #print (_x-last_x, _y-last_y)
      last_x = _x ; last_y = _y ;

    return auc ;


  #ax.plot(x[indices],y[indices]) ;
   


if __name__=="__main__":
  parser =  ArgumentParser() ;

  parser.add_argument("--json", type=str, required=True)
  parser.add_argument("--make_plots", type=eval, required=False,default=False)

  FLAGS = parser.parse_args(sys.argv[1:]) ;

  oc = OutlierComputation(**vars(FLAGS)) ;
  print("AUC is", oc.compute_auc()) ;
  oc.create_plots() ;





