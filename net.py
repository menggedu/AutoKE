import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *
from torch.nn import Linear,Tanh,Sequential,ReLU,Softplus
import logging
# CUDA support
import pandas as pd
from parse import *
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
    
class PINN(torch.nn.Module):
    def __init__(self, layers,lb,ub,device):
        super(PINN, self).__init__()
        self.u_model = torch.nn.ModuleList()
        self.u_model.append(Linear(layers[0], layers[1]))
        for width in layers[2:-2]:
            self.u_model.append(Tanh())
            self.u_model.append(Linear(width, width))
   
        self.u_model.append(Tanh())
        self.u_model.append(Linear(layers[-2], layers[-1]))
    
        self.ub= torch.from_numpy(ub).float().to(device)
        self.lb = torch.from_numpy(lb).float().to(device)
    
        self.init_weights(self.u_model)

        
    def forward(self,X):
        # X= 2*(X-self.lb)/(self.ub-self.lb)-1
        X= (X-self.lb)/(self.ub-self.lb)
        for layer in self.u_model:
            X = layer(X)
        return X
    
    def init_weights(self,m):
        if type(m) == Linear:
            torch.nn.init.xavier_uniform(m.weight)
            

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation,lb,ub, device,use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        self.ub= torch.from_numpy(ub).float().to(device)
        self.lb = torch.from_numpy(lb).float().to(device)
        
    def forward(self, x):
        x= (x-self.lb)/(self.ub-self.lb)
        out = self.layers(x)
        return out

class PhysicsInformedNN():
    """PINN network"""
    def __init__(self, args, X_star,u_star, X_init, u_init, X_f_train, bc_lb, bc_ub, layers, nu, beta, rho, optimizer_name, lr,
         lb=None,ub=None,activation='tanh',
        auto=False, 
        pretrained_model_path = None):

        self.system = args.system
        self.epoch_adam = args.epoch_adam
        self.X_star = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
        self.X_t = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
        self.u_star = u_star
        # init
        self.x_init = torch.tensor(X_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(X_init[:, 1:2], requires_grad=True).float().to(device)
        self.u_init = torch.tensor(u_init, requires_grad=True).float().to(device)
        # colloc
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        #boundary
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.u_bc_lb = torch.tensor(bc_lb[:, 2:3], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u_bc_ub = torch.tensor(bc_ub[:, 2:3], requires_grad=True).float().to(device)

        self.dnn = DNN(layers, activation, lb,ub,device).to(device)

        if pretrained_model_path is not None:
            # load pretrained model 
            print(f"loading model from {pretrained_model_path}")
            self.load_pretrained_model(pretrained_model_path)

        
        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.lr = lr
        self.optimizer = choose_optimizer(optimizer_name, self.dnn.parameters(), self.lr)
        self.optimizer2 = choose_optimizer("Adam", self.dnn.parameters())

        self.auto = auto
        self.iter = 0
        self.loss_list = []
        self.l1_list = []
        self.l2_list = []
        
        if self.system == 'vgc':
            e_ascii = 'diff(F2,x) - diff(u,t)'
        elif self.system == 'kdv':
            e_ascii = '1 * u * diff(u,x) + 0.0025 * diff(u,x,3) + diff(u,t)'
        else:
            raise NotImplementedError
            print(f'new sytstem: {self.system}')
            # define e_ascii
            e_ascii = ''
            print(f"ascii representation: {e_ascii}")
        if self.auto:
            lexer = Lexer(e_ascii)
            tokens = lexer.generate_tokens()
            parser = Parser(tokens)
            self.ast = parser.parse()
        
        

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u
    
    def load_pretrained_model(self, path):
        state_dict = torch.load(path)
        if "module" in state_dict:
            state_dict = state_dict['module']
        self.dnn.load_state_dict(state_dict)
        print(f"load model from {path}")
        
    
    def net_f(self, x, t):
        """ 
        Manually implement physics constraint:
        Autograd for calculating the residual for different systems.
        """
        
        u = self.net_u(x, t)
        print(u)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
        u_xxx = torch.autograd.grad(
            u_xx, x,
            grad_outputs=torch.ones_like(u_xx),
            retain_graph=True,
            create_graph=True
            )[0]

        if self.system == 'rd':
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif self.system == 'kdv':
            f = u_t + u*u_x+0.0025*u_xxx
        elif self.system == 'vgc':
            compound = u_x*u
            compound_x = torch.autograd.grad(
                compound,x,
                grad_outputs = torch.ones_like(compound),
                retain_graph=True,
                create_graph = True
            )[0]
            
            f = u_t-compound_x

        return f
    
    
    def autocal_residual(self,x,t):
        self.u = self.net_u(x, t)
        # self.u = self.net_u(self.x_f, self.t_f)
        variable_set = {
            'u':self.u,
            'x':self.x_f,
            't':self.t_f
        }
        def traverse(node):
            if node.token_type == TokenType.T_VARIABLE :
                return variable_set[node.value]
                
            if node.token_type ==  TokenType.T_NUM:
                return eval(node.value)
                
            left_result = traverse(node.children[0])
            right_result = traverse(node.children[1])
            operation = operations[node.token_type]
            
            result = operation(left_result, right_result)
            return result
        # residual = compute(self.ast)
        residual = traverse(self.ast)
        
        return residual
        
    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x

    def loss_pinn(self, verbose=True):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        
        u_pred_init = self.net_u(self.x_init, self.t_init)
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
        
        if not self.auto:
            f_pred = self.net_f(self.x_f, self.t_f)
            
        else:
            f_pred = self.autocal_residual(self.x_f,self.t_f)
        
        loss_u = torch.mean((self.u_init - u_pred_init) ** 2)
        
        if self.system == 'rd':
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        elif self.system == 'kdv'or self.system == 'vgc':
            loss_b = torch.mean((self.u_bc_lb - u_pred_lb)**2)
            loss_b+= torch.mean((self.u_bc_ub - u_pred_ub)**2)
        else:
            pass
        loss_f = torch.mean(f_pred ** 2)

        #default weights = 1
        loss = loss_u + loss_b + loss_f

        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_b.item(), loss_f.item())
                )
            self.iter += 1
        self.loss_list.append(loss.data.cpu().item())
        return loss

    def train(self):
        
        self.dnn.train()
        for i in range(self.epoch_adam):
            loss = self.loss_pinn()
            self.optimizer2.step()
            if (i+1)%1000 == 0 :
                self.test()
            # self.optimizer2.step(self.loss_pinn)
        print("lfgs start")
        self.iter=0
        self.optimizer.step(self.loss_pinn)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()
        return u
    
    def test(self):
        self.dnn.eval()
        u = self.net_u(self.X_star, self.X_t)
        u_pred = u.detach().cpu().numpy()
        self.dnn.train()
        error_u_relative = np.linalg.norm(self.u_star-u_pred, 2)/np.linalg.norm(self.u_star, 2)
        error_u_abs = np.mean(np.abs(self.u_star - u_pred))
        error_u_linf = np.linalg.norm(self.u_star - u_pred, np.inf)/np.linalg.norm(self.u_star, np.inf)
        self.l2_list.append(error_u_relative)
        self.l1_list.append(error_u_abs)
        
    def save_mid_res(self,path):
        "middle results"
        error = {
            'l1_error': self.l1_list,
            'l2_error': self.l2_list,
        }
        loss = {
            'loss': self.loss_list
        }
        result = pd.DataFrame.from_dict(error)
        loss_res = pd.DataFrame.from_dict(loss)
        result.to_csv(path+'/error.csv')
        loss_res.to_csv(path+'/loss.csv')
        
