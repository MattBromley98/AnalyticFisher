import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

def WeightTransfer(refModel, newModel):
    """
    This function handles transfering pretrained weights of a PyTorch model onto a fresh PyTorch model so that a certain layer may be randomised 
    """
    newModel.load_state_dict(refModel.state_dict())
    #Delete the old Model
    del(refModel)
    return newModel;

def jacobian(f, w, create_graph=False):   
    """
    function to find the jacobian of f with respect to x parameters of model
    output has shape (len(f), len(x))
    """
    jac = []
    output = f
    #f = torch.log(f)
    grad_f = torch.zeros_like(f)
    for i in range(len(f)):                                                                      
        grad_f[i] = 1.
        grad_f_x = torch.autograd.grad(f, w, grad_f, retain_graph=True, create_graph=create_graph, allow_unused=True)
        J = torch.cat(grad_f_x).view(-1)
        jac.append(J * torch.sqrt(output[i]))
        grad_f[i] = 0.
    return torch.stack(jac).reshape(f.shape + J.shape)

def CalcFIM(net, train_loader, n_iterations, weightname, approximation=None, weight_init="uniform"):
    """
    Main Fisher Analysis Initialisation
    net-> Compatible PyTorch Model
    train_loader -> PyTorch/Tensorflow/SKLearn Compatible Train Loader
    n_iterations -> Number of Iterations to Calculate the Fisher For
    weightname -> Name of the Layer in model.named_parameters()
    approximation -> default=None -> change to "emperical" to use the Emperical Method of Fisher Calculation
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Rank = []
    FR = []
    print(f"Calculating {n_iterations} of the Fisher...")
    Number_of_FisherIts = n_iterations
    
    #Obtain the number of weights in the chosen layer
    for name, param in net.named_parameters():
        if name==weightname:
            n_weight = len(param)
            if(weight_init == "uniform"):
                torch.nn.init.uniform_(param, -1., 1.)
            if(weight_init == "normal"):
                torch.nn.init.normal_(param, mean=0.0, std=1.0)
            if(weight_init == "xavier"):
                torch.nn.init.xavier_uniform_(param, gain=1.0)
            if(weight_init == "kaiming"):
                torch.nn.init.kaiming_normal(param)
            if(weight_init == "orthogonal"):
                torch.nn.init.orthogonal_(param, gain=1.0)
        
    realisations_torch = torch.zeros((Number_of_FisherIts,n_weight,n_weight))
    for i in tqdm(range(Number_of_FisherIts)):
        #Perform the appropriate weight initilisation
        for name, param in net.named_parameters():
            if(name==weightname):
                w = param
        flat_w = w.view(-1).cpu()
        fisher = torch.zeros((n_weight,n_weight)).to(device)
        for x_n, y_n in train_loader:
            x_n, y_n = x_n.to(device), y_n.to(device)
            f_n = net(x_n)
            summedfisher = np.zeros((Number_of_FisherIts, n_weight, n_weight))
            for row in f_n:
                if (approximation=="emperical"):
                    pi_n = F.softmax(row, dim=0)
                    diag_pi_n = torch.diag(pi_n.squeeze(0)).to(device)
                    pi_n_pi_n_T=torch.from_numpy(np.outer(pi_n.cpu().detach().numpy(),np.transpose(pi_n.cpu().detach().numpy()))).to(device)
                #J_f = get_gradient(row, w)
                J_f = jacobian((torch.squeeze(row,0)),w)
                jacob1 = J_f.cpu()
                if (approximation=="emperical"):   
                    J_f_T = J_f.permute(1,0)
                    K2 = diag_pi_n-pi_n_pi_n_T
                    K3 = K2.cuda()
                    fisher += torch.matmul((torch.matmul(J_f_T,K3)),J_f)
                else:
                    temp_sum = np.zeros((2, n_weight, n_weight))
                    grads = jacob1.detach().numpy()
                    for j in range(2):
                        temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
                    summedfisher[i] += np.sum(temp_sum, axis=0)
                    fisher += torch.from_numpy(summedfisher[i]).to(device)
            with torch.no_grad():
                try:
                    rank = torch.matrix_rank(fisher).item()
                    Rank.append(rank)
                    realisations_torch[i] = fisher.cpu()
                    Fw = np.matmul(fisher.cpu().numpy(),flat_w)
                    wFw = np.dot(flat_w,Fw)
                    FR.append(wFw)
                except:
                    pass
    return realisations_torch, Rank, FR;

def Hessian(y, x):
    """
    Function that computes the Hessian with respect to x
    """
    hessian = Jacobian(Jacobian(y, x, create_graph=False), x)
    return hessian;

def calc_eig(Fishers):
    """
    Function that computes Eigenvalues of the Fisher Information Matrix
    """
    for fisher in Fishers:
        eigval = torch.eig(fisher, eigenvectors=False,  out=None).eigenvalues[:,0]
    return eigval

def normalise(Fishers):
    """
    Function that normalises a Tensor of multiple Fisher Matrices
    """
    fisher_trace = np.trace(np.average(Fishers, axis=0)) 
    finalfisher = np.average(Fishers, axis=0)
    fhat = 12 * finalfisher/fisher_trace
    return fhat

def effective_dimension(model, f_hat, num_thetas, n, outputs):
    '''
    Function that computes an APPROXIMATION of the effective dimension at different values of the number of samples as computed in https://zenodo.org/record/4732830 by Amira Abbas
    model -> pytorch compatible model
    f_hat -> normalised array of fisher matrix
    n -> Number of samples for the computation (len trainloader?) in list
    outputs -> Number of classes
    '''
    effective_dim = []
    for ns in tqdm(n):
        Fhat = f_hat * ns / (2 * np.pi * np.log(ns))
        one_plus_F = np.eye(outputs) + Fhat
        det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
        r = det / 2  # divide by 2 because of sqrt
        effective_dim.append(2 * (logsumexp(r) - np.log(num_thetas)) / np.log(ns / (2 * np.pi * np.log(ns))))
    return effective_dim