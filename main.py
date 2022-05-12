"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""

import argparse
from net import *
import numpy as np
import os
import random
import torch
from data_pro import pre_data_all
from systems import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt
import logging
################
# Arguments
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main():
    parser = argparse.ArgumentParser(description='AutoKE')

    parser.add_argument('--system', type=str, default='rd', choices = ['rd','kdv','vgc'], help='System to study.')
    parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
    parser.add_argument('--N_f', type=int, default=10000, help='Number of collocation points to sample.')
    parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
    parser.add_argument('--lr', type=float, default=2.0, help='Learning rate.')


    parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
    parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
    parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
    parser.add_argument('--rho', type=float, default=3.0, help='reaction coefficient for u*(1-u) term.')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
    parser.add_argument('--u0_str', default='gauss', help='str argument for initial condition if no forcing term.')
    parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

    parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
    parser.add_argument('--epoch-adam', default=10000, type=int,  help='Loss for the network (MSE, vs. summing).')
    parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')


    parser.add_argument('--visualize', default=True, help='Visualize the solution.')
    parser.add_argument('--save_model', default=True, help='Save the model for analysis later.')
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--name', default='rd')
    parser.add_argument('--mode', default='train',type = str,choices=['train','test','pretrain'], help = 'mode for the NN' )
    parser.add_argument('--auto', action = 'store_true', help = 'automatically implement physical constraints' )
    parser.add_argument('--pretrain-model-path', default=None,type = str,  help = 'checkpoint path for pretraining' )

    args = parser.parse_args()
    train(args)
        
def train(args):
     # CUDA support
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    set_seed(0)
    nu = args.nu
    beta = args.beta
    rho = args.rho

    if args.system == 'rd': # reaction-diffusion
        beta = 0.0
        print('nu', nu, 'beta', beta, 'rho', rho)

    # parse the layers list here
    orig_layers = args.layers
    
    ############################
    # Process data
    ############################

    # X_star,u_star, X_init, u_init, X_f_train, bc_lb, bc_ub, lb, ub
    X_star, u_star, X_f_train, X_init, u_init,bc_lb, bc_ub,lb,ub,x,t = pre_data_all(args)
    layers = [int(item) for item in args.layers.split(',')]
    set_seed(args.seed) # for weight initialization

    model = PhysicsInformedNN(args, X_star,u_star, X_init, u_init, X_f_train, bc_lb, bc_ub, layers, nu, beta, rho,
                                args.optimizer_name, args.lr, lb,ub,args.activation,args.auto, args.pretrain_model_path)
    
    path =f"result/{args.system}/{args.name}"
    ############################
    # Train the model
    ############################
    if args.mode == 'train':
        model.train()
    elif args.mode == 'test':
        pass
        #model.load_pretrained_model(f"{path}/{args.name}.pt")
    else:
        print("Unknown mode is not included")
        assert False

    u_pred = model.predict(X_star)

    error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    error_u_abs = np.mean(np.abs(u_star - u_pred))
    error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)

    print('Error u rel: %e' % (error_u_relative))
    print('Error u abs: %e' % (error_u_abs))
    print('Error u linf: %e' % (error_u_linf))

    if args.visualize:
        if not os.path.exists(path):
            os.makedirs(path)
        model.save_mid_res(path)
        u_pred = u_pred.reshape(len(t), len(x))
        plot_u(u_star,x,t, args.name,args.system, path)
        plot_u(u_pred, x,t, args.name,args.system, path, data_type='Prediction')
        plot_diff(u_star, u_pred, x,t, args.name, args.system, path)
        x_u_train = np.vstack((bc_lb[:,:2],bc_ub[:,:2],X_init))
        u_train = np.vstack((bc_lb[:,2:],bc_ub[:,2:],u_init))
        if args.system == 'vgc':
            plot_solutions_vgc(X_star, u_pred,x_u_train, u_train, u_star, x,t,path)
        else:
            plot_solutions(X_star, u_pred,x_u_train, u_train, u_star, x,t,path)
            
    if args.save_model and args.mode == 'train': # whether or not to save the model
        state_dict = {}
        state_dict['module'] = model.dnn.state_dict()
        torch.save(state_dict, f"{path}/{args.name}.pt")

def test(args):
    pass

if __name__ == '__main__':
    main()
    