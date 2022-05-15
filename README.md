# AutoKE
The PyTorch Source code for [An automatic knowledge embedding framework for scientific machine learning](https://doi.org/10.48550/arXiv.2205.05390).



## Instructions
To run the code for conducting the experiments in the paper, follow the script and optional parameters below:
```
python main.py [--system] [--seed] [--N_f] [--optimizer_name] [--lr] [--xgrid] [--nt] [--u0_str] [--layers] [--epoch-adam] [--activation][--visualize] [--save_model] [--data-path] [--name] [--mode] [--auto] [--pretrain-model-path]

Possible arguments:
--system              system of study (default:rd; also support kdv equation, viscous gravity current(vgc), and new system defined by yourself)
--seed                used to reproduce the results (default: 0)
--N_f                 number of points to sample from the interior domain (default: 1000)
--optimizer_name      optimizer to use, currently supports L-BFGS
--lr                  learning rate (default: 1.0)
--xgrid               size of the xgrid (default: 256)
--nt                  Number of points in the tgrid (default 100)
--u0_str              initial condition (default: 'sin(x)'; also supports 'gauss' for reaction/reaction-diffusion)
--layers              number of layers in the network (default: '50,50,50,50,1')
--epoch-adam          epoch numbers for adam optimizer (default:10000)
--activation          activation for the network (default: 'tanh')
--visualize           option to visualize the solution (default: False)
--save_model          option to save the model (default: False)
--data-path           path of data
--name                job name 
--mode                mode of the NN (default: traing; also support "test"  and "pretrain")
--auto                option to implement physical constraints automatically (default: False)
--pretrain-model-path checkpoint path for pretraining
```

## Example
For solving the KdV equation, you need to write a JSON format configuration file in accordance with the prescribed format. See `kdv.json` in folder `config/`. It contains the system name, ASCII representation of the equaiton, and some subcomponents for the equation.

Training scripts:
```
python main.py --system kdv --layers 2,128,128,128,128,128,128,1 --mode train --name kdv --epoch-adam 20000 --auto  
```

## To be updated:
New features and cases

# Reference resources
https://github.com/tensordiffeq/TensorDiffEq  
https://github.com/a1k12/characterizing-pinns-failure-modes

