import scipy.io
import numpy as np
# import tensorflow as tf
import time
from datetime import datetime
from pyDOE import lhs
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


from systems import *
from net import *
from utils import *
from plotting import newfig, savefig


def pre_data_all(args,):
          
    if args.system == 'kdv':
        data = scipy.io.loadmat(args.data_path+'/kdv/kdv.mat')
        
        t = data['tt'].flatten()[:,None] # T x 1
        x = data['x'].flatten()[:,None] # N x 1
        Exact = np.real(data['uu']).T # T x N    
        u_star = Exact.reshape(-1,1)  
        
    elif args.system == 'rd':
        u_vals = reaction_diffusion_discrete_solution(args.u0_str, args.nu, args.rho, args.xgrid, args.nt)
        x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, args.nt).reshape(-1, 1)
        u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
        Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid
        
    elif args.system == 'vgc':
        data = np.load(args.data_path+'/vgc/vgc.npy')
        nx = 100
        nt = 251
        x=np.linspace(1,2,nx).reshape(-1,1)
        t=np.linspace(0,0.5,nt).reshape(-1,1)
        Exact = data
        u_star = Exact.reshape(-1,1)
    
    else:
        print("WARNING: System is not specified.")   
 
    # 
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)
    print(lb,ub)
    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    
    x_noboundary = x[1:]# remove boundary at x=0
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    # X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)
    # import pdb;pdb.set_trace()
    X_f_train = lb + (ub-lb)*lhs(2, args.N_f)
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    
    
    uu2 = Exact[:,0:1] # u(-end, t)
    bc_lb = np.hstack((X[:,0:1], T[:,0:1], uu2)) # boundary condition at x = 0, and t = [0, 1]
    # generate the other BC, now at x=2pi
    
    x_bc_ub = X[:,-1:]
    if args.system == 'rd':
      x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    uu3 = Exact[:,-1].reshape(-1,1)
    # import pdb;pdb.set_trace()
    bc_ub = np.hstack((x_bc_ub, t,uu3))
    

    u_init = uu1 # just the initial condition
    X_init = xx1 # (x,t) for initial condition

    return X_star, u_star, X_f_train, X_init, u_init,bc_lb, bc_ub,lb,ub,x,t

  
class Logger(object):
  def __init__(self, frequency=10):
    pass

    self.start_time = time.time()
    self.frequency = frequency

  def __get_elapsed(self):
    return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

  def __get_error_u(self):
    return self.error_fn()

  def set_error_fn(self, error_fn,error_fn2=None):
    self.error_fn = error_fn
    self.error_fn2 = error_fn2
    
  def log_train_start(self, model):
    print("\nTraining started")
    print("================")
    self.model = model
    print(self.model.summary())

  def log_train_epoch(self, epoch, loss,custom="", is_iter=False):
    if epoch % self.frequency == 0:
      error, pde_loss = self.__get_error_u()
      print(f"{'nt_epoch' if is_iter else 'epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  error = {error:.4e} pde_loss = {pde_loss:.4e}  " + custom)
  
  def log_train_epoch2(self, epoch, loss,num=0,custom="", is_iter=False):
    if epoch % self.frequency == 0:
      error, pde_loss = self.error_fn2(num)
      print(f"{'nt_epoch' if is_iter else 'epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  error = {error:.4e} pde_loss = {pde_loss:.4e}  " + custom)

  def log_train_opt(self, name):
    # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
    print(f"—— Starting {name} optimization ——")

  def log_train_end(self, epoch, custom=""):
    print("==================")
    error, pde_loss = self.__get_error_u()
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  error = {error:.4e}  " + custom)
