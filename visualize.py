"""
Visualize outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import newfig, savefig

def plot_image():
    pass
def plot_u(Exact, x, t,name, system, path,data_type='Exact', file_type = 'png'):
    """Visualize exact solution."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    Exact = Exact.reshape(len(t), len(x))
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=20)
    ax.set_ylabel('x', fontweight='bold', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/{data_type}_{name}_system_{system}.{file_type}")
    plt.close()

    return None

def plot_diff(Exact, U_pred, x, t, name, system, path, file_type = 'png',relative_error = False):
    """Visualize abs(u_pred - u_exact)."""

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    Exact = Exact.reshape(len(t), len(x))
    U_pred = U_pred.reshape(len(t), len(x))
    if relative_error:
        h = ax.imshow(np.abs(Exact.T - U_pred.T)/np.abs(Exact.T), interpolation='nearest', cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(np.abs(Exact.T - U_pred.T), interpolation='nearest', cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    # cbar.set_ticks()
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=20)
    ax.set_ylabel('x', fontweight='bold', size=20)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/udiff_Exact_{name}_system_{system}_.{file_type}")

    return None

def u_predict(u_vals, U_pred, x, t, nu, beta, rho, seed, layers, N_f, L, source, lr, u0_str, system, path):
    """Visualize u_predicted."""

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    # colorbar for prediction: set min/max to ground truth solution.
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=u_vals.min(0), vmax=u_vals.max(0))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)

    plt.savefig(f"{path}/upredicted_{system}_nu{nu}_beta{beta}_rho{rho}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf")

    plt.close()
    return None

def plot_solutions_vgc(X_star, u_pred, X_u_train, u_train, Exact_u, x, t, path):
    
    # Interpolating the results on the whole (x,t) domain.
    # griddata(points, values, points at which to interpolate, method)
    
    X,T = np.meshgrid(x,t)
    U_pred = u_pred
    # u_pred = u_pred.flatten()
    # U_pred = griddata(X_star, u_pred, (X, T), method='cubic')
    u_max = Exact_u.max()
    u_min = Exact_u.min()
    Exact_u = (Exact_u-u_min)/(u_max-u_min)
    U_pred = (U_pred-u_min)/(u_max-u_min)
    Exact_u = Exact_u.reshape(len(t), len(x))

    # Creating the figures
    fig, ax = newfig(1.2, 1.8)
    ax.axis('off')
    low = int(0.25*len(t))
    mid = int(0.5*len(t))
    high = int(0.75*len(t))
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(2, 1)
    gs0.update(top=1-0.06, bottom=1-3/5, left=0.1, right=0.9, wspace=0.2)
    
        
    ax = plt.subplot(gs0[0, :])

    h = ax.imshow(Exact_u.T, interpolation='nearest', #cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto',vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('Exact', fontsize = 12)
    ax.set_xticks(np.arange(0,0.5,0.1) )
    ax.set_xticklabels([])
    # ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax = plt.subplot(gs0[1, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', #cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto',vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    # ax.plot(X_f[:,1], X_f[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 1, clip_on = False)

    # line = np.linspace(x.min(), x.max(), 2)[:,None]
    # ax.plot(t[low]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[mid]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[high]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    # ax.legend(frameon=False, loc = 'best')
    ax.set_title('Prediction', fontsize = 12)

    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-3/5, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    # import pdb;pdb.set_trace()
    ax.plot(x,Exact_u[low,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[low,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.12$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([0.9,2.1])
    ax.set_ylim([-0.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_u[mid,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[mid,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.24$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([0.9,2.1])
    ax.set_ylim([-0.1,1.1])

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_u[high,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[high,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([0.9,2.1])
    ax.set_ylim([-0.1,1.1])

    ax.set_title('$t = 0.36$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5,fontsize = 12, frameon=False)
    
    plt.savefig(f'{path}_solution.png',dpi = 500)   

def plot_solutions(X_star, u_pred, X_u_train, u_train, Exact_u, x, t, path):
    
    # Interpolating the results on the whole (x,t) domain.
    # griddata(points, values, points at which to interpolate, method)
    # import pdb;pdb.set_trace()
    X,T = np.meshgrid(x,t)
    U_pred = u_pred
    # u_pred = u_pred.flatten()
    # U_pred = griddata(X_star, u_pred, (X, T), method='cubic')
    u_max = Exact_u.max()
    u_min = Exact_u.min()
    Exact_u = (Exact_u-u_min)/(u_max-u_min)
    U_pred = (U_pred-u_min)/(u_max-u_min)
    Exact_u = Exact_u.reshape(len(t), len(x))

    # Creating the figures
    fig, ax = newfig(1.4, 1.8)
    ax.axis('off')
    low = int(0.25*len(t))
    mid = int(0.5*len(t))
    high = int(0.75*len(t))
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2, left=0.1, right=0.9, wspace=0.2)
    
        
    ax = plt.subplot(gs0[:, 0])

    h = ax.imshow(Exact_u.T, interpolation='nearest', #cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_title('Exact', fontsize = 12)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax = plt.subplot(gs0[:, 1])

    h = ax.imshow(U_pred.T, interpolation='nearest', #cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    # ax.plot(X_f[:,1], X_f[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 1, clip_on = False)

    # line = np.linspace(x.min(), x.max(), 2)[:,None]
    # ax.plot(t[low]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[mid]*np.ones((2,1)), line, 'w-', linewidth = 1)
    # ax.plot(t[high]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    # ax.set_ylabel('$x$')
    # ax.legend(frameon=False, loc = 'best')
    ax.set_title('Prediction', fontsize = 12)
    ax.set_yticklabels([])
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/2, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    # import pdb;pdb.set_trace()
    ax.plot(x,Exact_u[low,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[low,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_u[mid,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[mid,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])


    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_u[high,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[high,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])

    ax.set_title('$t = 0.75$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize = 12, frameon=False)
    
    plt.savefig(f'{path}_solution.png',dpi = 500)   

if __name__ == '__main__':
    from pylab import *
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    f = np.arange(0,101)                 # frequency 
    t = np.arange(11,245)                # time
    z = 20*np.sin(f**0.56)+22            # function
    z = np.reshape(z,(1,max(f.shape)))   # reshape the function
    Z = z*np.ones((max(t.shape),1))      # make the single vector to a mxn matrix
    T, F = meshgrid(f,t)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(F,T,Z)
    plt.xlim((t.min(),t.max()))
   
    cbar=plt.colorbar()              # the mystery step ???????????
    # cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1]) # add the labels
    plt.show()