"""
This code is to compare PDASGD(our algorithm) with AAM, APDAGD,PDASMD,Sinkhorn, stochastic Sinkhorn  on MNIST data. 


Some parts are modified by https://github.com/PythonOT/POT/tree/master/ot, https://github.com/JasonAltschuler/OptimalTransportNIPS17 and https://github.com/nazya/AAM. 
"""
########################################
# Import the packages and MNIST dataset#
########################################
from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
from multiprocess import Pool, cpu_count
import multiprocess
#https://pypi.org/project/python-mnist
#first download MNIST data and change the path to directory of MNIST
mndata = MNIST()
images, labels = mndata.load_training()
import statistics 
from scipy.special import softmax
import scipy.special
from PIL import Image


#%%
##########################################################################
# The algortithms: PDASGD/AAM/APDAGD/Sinkhorn/PDASMD/stochastic Sinkhorn # 
##########################################################################
#%% PDASGD(our algorithm)
def grad(v, eta, C, alpha, beta, num, Size):  
    Cnum = C[num,]
    Cnum = np.reshape(Cnum, (-1,1))
    etavec = np.repeat(eta, Size).reshape(-1,1)
    tempx = (v - Cnum - etavec) / eta 
    return (   -Size * alpha[num] * ( beta- softmax(tempx) )   )


def fullgrad(v, eta, C, alpha, beta, Size): 
    full=-Size*(alpha*(beta.T-softmax( (v.T-C-eta) /eta, axis=1)))
    return full


def primal_semidual(semi, alpha, beta, C, Size, eta):
     semi = np.reshape(semi,(Size,-1))
     p = (softmax( (semi.T-C-eta) /eta, axis=1)*alpha).T.reshape(-1,1)
     return p

#the main algorithm
def PDASGD(alpha, beta, inner_iters, Size, acc, penalty, C):
      seed = 123
      np.random.seed(seed) 
      tau2 = 0.5
      s=0 
      y_store = np.zeros((Size, 1),np.float64)
      z_temp = np.zeros((Size, 1),np.float64)
      v_temp = np.zeros((Size, 1),np.float64) 
      vtilde = np.zeros((Size, 1),np.float64)
      x_s = np.zeros((Size, Size),np.float64)
      C_temp = 0
      D_temp = np.zeros((Size * Size, 1),np.float64)
      Lip = (1 / penalty)
      
      flag = 0 
      Store_Error = []
      Store_Flag = []
      Store_value = []
      acc_temp = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
      valueotinitial = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
      Store_Error.append(acc_temp)
      Store_Flag.append(0)
      Store_value.append(valueotinitial)
      error=100
      
      while error >= acc: 

          tau1 = 2 / (s + 4)
          flag = flag + 2
          
          alphas = 1 / (9 * tau1 * Lip)
          flag =  flag + 3
          
          recorded_gradients = fullgrad(v = vtilde, eta = penalty, C = C, alpha = alpha, beta = beta, Size = Size)
          flag = flag + 9 * Size * Size -  Size
          
          full_gradient = recorded_gradients.mean(axis=0).reshape(-1,1) 
          flag = flag + Size * Size
          
          store_y=[]
          for _ in range(inner_iters):      
                j = np.random.multinomial(1, alpha.reshape(-1)).argmax()
                v_temp = tau1 * z_temp + tau2 * vtilde + (1 - tau1 - tau2) * y_store
                flag = flag + 5 * Size + 2
                
                grad_temp = full_gradient + ( grad(v = v_temp, eta = penalty, C = C, alpha = alpha, beta = beta, num = j, Size = Size) - recorded_gradients[j].reshape(-1,1) ) / (alpha[j]*Size)
                flag = flag + 8 * Size 
                flag = flag + Size
                
                z_temp = z_temp - alphas * grad_temp * 15
                flag = flag + 2 * Size
                
                y_store = v_temp- grad_temp / (9 * Lip)
                flag = flag + 2 * Size + 1
                
                store_y=np.append(store_y,y_store) 
          store_y = np.reshape(store_y,(-1,Size)) 
          vtilde = store_y.mean(axis = 0).reshape(-1,1)
          flag = flag + inner_iters
          
          C_temp = C_temp + 1 / tau1
          flag = flag + 2
          
          t = np.random.choice(inner_iters)
          random_y = store_y[t,:].reshape(-1,1)
          D_temp = D_temp + (1 / tau1) * primal_semidual(random_y, alpha = alpha, beta = beta, C =C, Size = Size, eta = penalty) 
          flag = flag + 7 * Size * Size - Size
          flag = flag + Size * Size + Size * Size +1
          
          x_s = (D_temp/C_temp).reshape(Size, Size).T #convert to matrix
          flag = flag + Size * Size
          
          s = s + 1 
          
          
          
          error = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
          Store_Error.append(error)
          Store_Flag.append(flag)

          
          valueot = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
          Store_value.append(valueot)
          
                    
          
      return x_s, flag
  
def roundto(F, alpha, beta): #round to feasible area
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G
     
    
def approxtoPDASGD(epsilon, Size, inner_iters, numberC, numberdistribution):
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter = PDASGD(alpha = alphatilde, beta = betatilde , inner_iters = inner_iters, Size = Size, 
           acc = epsilonprime / 2, 
           penalty = penalty, C = C)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, Iter

#%% PDASMD
def grad(v, eta, C, alpha, beta, num, Size):  
    Cnum = C[num,]
    Cnum = np.reshape(Cnum, (-1,1))
    etavec = np.repeat(eta, Size).reshape(-1,1)
    tempx = (v - Cnum - etavec) / eta 
    return (   -Size * alpha[num] * ( beta- softmax(tempx) )   )


def fullgrad(v, eta, C, alpha, beta, Size): 
    full=-Size*(alpha*(beta.T-softmax( (v.T-C-eta) /eta, axis=1)))
    return full


def primal_semidual(semi, alpha, beta, C, Size, eta):
     semi = np.reshape(semi,(Size,-1))
     p = (softmax( (semi.T-C-eta) /eta, axis=1)*alpha).T.reshape(-1,1)
     return p

#the main algorithm
def PDASMD(alpha, beta, inner_iters, Size, acc, penalty, C):
      seed = 123
      np.random.seed(seed) 
      tau2 = 0.5
      s=0 
      y_store = np.zeros((Size, 1),np.float64)
      z_temp = np.zeros((Size, 1),np.float64)
      v_temp = np.zeros((Size, 1),np.float64) 
      vtilde = np.zeros((Size, 1),np.float64)
      x_s = np.zeros((Size, Size),np.float64)
      C_temp = 0
      D_temp = np.zeros((Size * Size, 1),np.float64)
      Lip = (1 / penalty)
      
      flag = 0 
      Store_Error = []
      Store_Flag = []
      Store_value = []
      acc_temp = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
      valueotinitial = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
      Store_Error.append(acc_temp)
      Store_Flag.append(0)
      Store_value.append(valueotinitial)
      error=100
      
      while error >= acc: 

          tau1 = 2 / (s + 4)
          flag = flag + 2
          
          alphas = 1 / (9 * tau1 * Lip)
          flag =  flag + 3
          
          recorded_gradients = fullgrad(v = vtilde, eta = penalty, C = C, alpha = alpha, beta = beta, Size = Size)
          flag = flag + 9 * Size * Size -  Size
          
          full_gradient = recorded_gradients.mean(axis=0).reshape(-1,1) 
          flag = flag + Size * Size
          
          store_y=[]
          for _ in range(inner_iters):      
                j = np.random.multinomial(1, alpha.reshape(-1)).argmax()
                v_temp = tau1 * z_temp + tau2 * vtilde + (1 - tau1 - tau2) * y_store
                flag = flag + 5 * Size + 2
                
                grad_temp = full_gradient + ( grad(v = v_temp, eta = penalty, C = C, alpha = alpha, beta = beta, num = j, Size = Size) - recorded_gradients[j].reshape(-1,1) ) / (alpha[j]*Size)
                flag = flag + 8 * Size 
                flag = flag + Size
                
                z_temp = z_temp - alphas * grad_temp / 2
                flag = flag + 2 * Size
                
                y_store = v_temp - np.linalg.norm(grad_temp, ord=1) / (9 * Lip) * np.sign(grad_temp)
                flag = flag + 4 * Size + 1
                
                store_y=np.append(store_y,y_store) 
          store_y = np.reshape(store_y,(-1,Size)) 
          vtilde = store_y.mean(axis = 0).reshape(-1,1)
          flag = flag + inner_iters
          
          C_temp = C_temp + 1 / tau1
          flag = flag + 2
          
          t = np.random.choice(inner_iters)
          random_y = store_y[t,:].reshape(-1,1)
          D_temp = D_temp + (1 / tau1) * primal_semidual(random_y, alpha = alpha, beta = beta, C =C, Size = Size, eta = penalty) 
          flag = flag + 7 * Size * Size - Size
          flag = flag + Size * Size + Size * Size +1
          
          x_s = (D_temp/C_temp).reshape(Size, Size).T #convert to matrix
          flag = flag + Size * Size
          
          s = s + 1 
          
          
          
          error = abs(x_s.sum(axis=1).reshape(-1,1) - alpha).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - beta).sum()
          Store_Error.append(error)
          Store_Flag.append(flag)

          
          valueot = np.dot(C.reshape(1,-1),x_s.reshape(-1,1))[0][0]
          Store_value.append(valueot)
          
          
          
          
      return x_s, flag
  
def roundto(F, alpha, beta): #round to feasible area
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G
     
    
def approxtoPDASMD(epsilon, Size, inner_iters, numberC, numberdistribution):
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter = PDASMD(alpha = alphatilde, beta = betatilde , inner_iters = inner_iters, Size = Size, 
           acc = epsilonprime / 2, 
           penalty = penalty, C = C)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, Iter



#%%APDAGD
def phi_(h, gamma, C, Size, conalpha, conbeta):
    one = np.ones(Size, np.float64)
    A = (-C/gamma + np.outer(h[:Size], one) + np.outer(one, h[Size:]))
    a = A.max()
    A-=a
    s = a+np.log(np.exp(A).sum())
    return gamma*(-h[:Size].dot(conalpha) - h[Size:].dot(conbeta) + s)

def f_(gamma, h, C, Size):
    y = (h.reshape(-1)).copy()
    y[h.reshape(-1) == 0.] = 1.
    y = y.reshape(Size, -1)
    return (C * h).sum() + gamma * (h * np.log(y)).sum()

def APDAGD(acc,conalpha,conbeta,C,gamma,Size):
    one = np.ones(Size, np.float64)
    L = 1.
    betta = 0.
    primal_var = 0.*np.ones([Size, Size], np.float64)
    z = np.zeros(2*Size, np.float64)
    z_new = np.zeros(2*Size, np.float64)
    eta = np.zeros(2*Size, np.float64)
    eta_new  = np.zeros(2*Size, np.float64)
    grad_phi_new = np.zeros(2*Size, np.float64)    
    f = lambda h: phi_(h, gamma=gamma, C=C, Size=Size, conalpha=conalpha, conbeta=conbeta)
    
    
    k=0
    oper_num=0
    
    
    K=-C/gamma
    oper_num = oper_num + Size * Size
    
    Store_Acc=[]
    Store_Oper=[]
    Store_value = []
    valueotinitial = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
    Store_value.append(valueotinitial)
    acc_temp = abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
    Store_Acc.append(acc_temp)
    Store_Oper.append(0)

    while True:
        L = L / 2
        oper_num = oper_num + 1
        while True:
            alpha_new = (1 + np.sqrt(4*L*betta + 1)) / 2 / L
            oper_num = oper_num + 7
            
            betta_new = betta + alpha_new   
            oper_num = oper_num + 1
            
            tau = alpha_new / betta_new
            oper_num = oper_num +1
            
            lamu_new = tau * z + (1 - tau) * eta   
            oper_num = oper_num + 2 * Size * 2 + 1                                 
            
            logB = (K + np.outer(lamu_new[:Size], one) + np.outer(one, lamu_new[Size:]))
            oper_num = oper_num + 4 * Size * Size
            
            max_logB =logB.max()
            logB_stable = logB - max_logB
            oper_num = oper_num + Size * Size
            
            B_stable = np.exp(logB_stable)
            oper_num = oper_num + Size * Size
            
            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            oper_num = oper_num + 2 * ( 2 * Size * Size - Size)
            
            Bs_stable = u_hat_stable.sum()
            oper_num = oper_num +  Size - 1
            
            phi_new = gamma*(-lamu_new[:Size].dot(conalpha) - lamu_new[Size:].dot(conbeta) + np.log(Bs_stable) + max_logB)
            oper_num = oper_num + 2*Size + 5
            
            grad_phi_new = gamma * np.concatenate((-conalpha + u_hat_stable/Bs_stable, -conbeta + v_hat_stable/Bs_stable),0)                       
            oper_num = oper_num + 6 * Size
                       
            z_new = z - alpha_new * grad_phi_new
            oper_num = oper_num + 4*Size     
            
            eta_new = tau * z_new + (1-tau) * eta
            oper_num = oper_num + 2*Size*3 + 1
            
            phi_eta = f(eta_new)
            oper_num = oper_num + 6*Size*Size+ 2*Size+4
            
            
            oper_num = oper_num + 3 + 10 *Size
            if phi_eta <= phi_new + grad_phi_new.dot(eta_new - lamu_new) + L * ((eta_new - lamu_new)**2).sum() / 2:
                
                betta = betta_new
                z = z_new.copy()
                eta = eta_new.copy()
                break    
            L = L * 2
            oper_num = oper_num + 1
          
        L= L/2
        oper_num = oper_num + 1
        
        primal_var = tau * B_stable/Bs_stable + (1 - tau) * primal_var
        oper_num= oper_num + 4*Size*Size + 1
        
        k=k+1
        


        valueot = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
        trueacc=abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
        Store_Acc.append(trueacc)
        Store_Oper.append(oper_num)
        Store_value.append(valueot)

          
        if  trueacc <= acc:  
            return primal_var, oper_num
           

def roundto(F, alpha, beta): 
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G
       
def approxtoAPDAGD(epsilon, Size, numberC,numberdistribution ):  
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, step= APDAGD(acc = epsilonprime / 2, conalpha = np.squeeze(alphatilde), conbeta = np.squeeze(betatilde), C = C, Size = Size, gamma= penalty)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, step
#%%Sinkhorn
def sinkhorn_log(a, b, M, reg, acc):
    dim_a = len(a)
    dim_b = b.shape[0]
    Size = dim_a


    oper = 0

    Mr = - M / reg
    oper += Size

    u = np.zeros(dim_a)
    v = np.zeros(dim_b)

    def get_logT(u, v):

        return Mr + u[:, None] + v[None, :]

    loga = np.log(a)
    logb = np.log(b)

    error = 1
    while True:

        v = logb - scipy.special.logsumexp(Mr + u[:, None], 0)
        oper += Size + Size * Size*3

        u = loga - scipy.special.logsumexp(Mr + v[None, :], 1)
        oper += Size + Size * Size*3

        G = np.exp(get_logT(u, v))
        oper += 3*Size*Size




        error = abs(np.sum(G, axis=1) - a).sum() + abs(np.sum(G, axis=0) - b).sum()
        if error < acc:
               return G, oper


def roundto(F, alpha, beta): #round to feasible area
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G
     
    
def approxtoSinkhorn(epsilon, Size, numberC, numberdistribution):
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter = sinkhorn_log(a = np.squeeze(alphatilde), b = np.squeeze(betatilde), M = C, reg = penalty, acc = epsilonprime / 2)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)   
    return Xhat, Iter

#%%Stochastic Sinkhorn
def stochasticsinkhorn(a, b, M, reg, acc):
    dim_a = a.shape[0]
    dim_b = b.shape[0]
    Size = dim_a

    oper = 0

    Mr = -M / reg
    oper += Size*Size

    K = np.exp(Mr)
    oper += Size*Size

    u = np.full((dim_a,), 1. / dim_a)
    oper += 1

    v = np.full((dim_b,), 1. / dim_b)
    oper += 1

    logu = np.log(u)
    oper += Size

    logv = np.log(v)
    oper += Size

    G = u[:, None] * K * v[None, :]
    oper += Size*Size*2

    G_log = np.exp(G)
    oper += Size*Size

    viol = np.sum(G, axis=1) - a
    oper += + Size*Size

    viol_2 = np.sum(G, axis=0) - b
    oper += + Size*Size



    while True:

        v = np.concatenate((viol,viol_2))
        p = np.abs(v)/np.abs(v).sum()
        j = np.random.multinomial(1, p.reshape(-1)).argmax()

        if j < Size:
            i_1 = j
            old_u_log = logu[i_1]

            new_u_log = np.log(a[i_1]) - scipy.special.logsumexp(Mr[i_1, :]+logv)
            oper += 1+1+Size+Size+Size


            G_log[i_1,:] = np.full(dim_a,new_u_log) +  Mr[i_1,:] + logv
            oper += Size*2


            viol[i_1]= np.exp(new_u_log + scipy.special.logsumexp(Mr[i_1, :]+logv))- a[i_1]
            oper += 1+1+1+Size+Size+Size

            viol_2 += np.exp(Mr[i_1, :].T + new_u_log + logv) - np.exp(Mr[i_1, :].T + old_u_log + logv)
            oper += Size*3 + Size + Size*3 + Size

            logu[i_1] = new_u_log
        else:
            i_2 = j -Size
            old_v_log = logv[i_2]

            new_v_log = np.log(b[i_2]) - scipy.special.logsumexp(Mr[:, i_2]+ logu)
            oper += 1+1+Size+Size+Size


            G_log[:,i_2] =   Mr[:,i_2] + logu + np.full(dim_a,new_v_log)
            oper += Size*2

            viol += np.exp(new_v_log + Mr[:, i_2] +logu) - np.exp(old_v_log + Mr[:, i_2] +logu)
            oper += 1+1+1+Size+Size+Size

            viol_2[i_2] = np.exp(new_v_log + scipy.special.logsumexp(Mr[:, i_2]+logu))- b[i_2]
            oper += Size*3 + Size + Size*3 + Size

            logv[i_2] = new_v_log

        G = np.exp(G_log) 
        ##We only need to update one column/one row here. But for convience, we implement to upadate all the elements. 
        ##When we calculate the number of operations, we only add n here.
        oper += Size
        error = abs(np.sum(G, axis=1) - a).sum() + abs(np.sum(G, axis=0) - b).sum()

        if error<=acc:
            return G,oper


def roundto(F, alpha, beta): #round to feasible area
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G


def approxtostochasticsinkhorn(epsilon, Size, numberC,numberdistribution):
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]   
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, Iter = stochasticsinkhorn(a = np.squeeze(alphatilde), b = np.squeeze(betatilde), M = C, reg = penalty, acc = epsilonprime / 2)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, Iter

#%%AAM
def phi_(h, gamma, C, Size, conalpha, conbeta):
    one = np.ones(Size, np.float64)
    A = (-C/gamma + np.outer(h[:Size], one) + np.outer(one, h[Size:]))
    a = A.max()
    A-=a
    s = a+np.log(np.exp(A).sum())
    return gamma*(-h[:Size].dot(conalpha) - h[Size:].dot(conbeta) + s)

def f_(gamma, h, C, Size):
    y = (h.reshape(-1)).copy()
    y[h.reshape(-1) == 0.] = 1.
    y = y.reshape(Size, -1)
    return (C * h).sum() + gamma * (h * np.log(y)).sum()

def AAM(acc, Size, gamma, conalpha, conbeta, C):
    L=1
    step = 2
    one = np.ones(Size, np.float64)
    x0 = np.zeros(2*Size, np.float64)
    xi = np.zeros_like(x0)
    eta = xi.copy()
    zeta = xi.copy()
    eta_new = xi.copy()
    zeta_new = xi.copy()
    grad2 = alpha_new = alpha = 0
    ustep = np.zeros_like(x0[:Size])
    vstep = np.zeros_like(ustep)
    
    f = lambda x: phi_(x, gamma, C, Size, conalpha, conbeta)
    f_primal = lambda x: f_(gamma, x, C, Size)
    
    
    operation = 0 
    

    
    primal_var = np.zeros_like(C)
    
    
    
    Store_Acc=[]
    Store_Oper=[]
    Store_value = []
    valueotinitial = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
    Store_value.append(valueotinitial)
    acc_temp = abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
    Store_Acc.append(acc_temp)
    Store_Oper.append(0)   

    
    while True:
        
        
        L_new = L/step
        operation = operation + 1
        
        K=-C/gamma
        operation = operation +Size *Size
        while True:
            alpha_new = 1/2/L_new + np.sqrt(1/4/L_new/L_new + alpha*alpha*L/L_new)
            operation = operation + 11
            
            tau = 1/alpha_new/L_new
            operation = operation + 2
            
            xi = tau * zeta + (1 - tau) * eta
            operation = operation + 6 * Size + 1
            
            
            ##############
            logB = (K + np.outer(xi[:Size], one) + np.outer(one, xi[Size:]))
            operation = operation + 4 * Size * Size
            
            max_logB =logB.max()
            logB_stable = logB - max_logB
            operation = operation + Size*Size

            B_stable = np.exp(logB_stable)
            operation = operation + Size*Size
            
            u_hat_stable, v_hat_stable = B_stable.dot(one), B_stable.T.dot(one)
            operation = operation + 2 * ( 2 * Size * Size - Size)
            
            
            Bs_stable = u_hat_stable.sum()
            operation = operation + Size - 1
            

            f_xi = gamma*(-xi[:Size].dot(conalpha) - xi[Size:].dot(conbeta) + np.log(Bs_stable) + max_logB)
            operation = operation + 2*Size + 5
            
            grad_f_xi = gamma*np.concatenate((-conalpha + u_hat_stable/Bs_stable, -conbeta + v_hat_stable/Bs_stable),0)            
            operation = operation + 6 * Size
            
            
            gu, gv = (grad_f_xi[:Size]**2).sum(), (grad_f_xi[Size:]**2).sum()
            operation = operation + 4*Size -2
            
            norm2_grad_f_xi = (gu+gv)
            operation = operation + Size

            if gu > gv:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        ustep = conalpha/u_hat_stable
                        operation = operation + Size
                        
                    except Warning as e:
                        u_hat_stable/=u_hat_stable.max()
                        operation = operation + Size
                        
                        u_hat_stable[u_hat_stable<1e-150] = 1e-150
                        ustep = conalpha/u_hat_stable
                        operation = operation + Size
                        #print('catchu')
                    
                
                ustep/=ustep.max()
                operation = operation + Size
                
                xi[:Size]+=np.log(ustep)
                operation = operation + Size *2
                
                Z=ustep[:,None]*B_stable
                operation = operation + Size
                
                
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        vstep = conbeta/v_hat_stable
                        operation = operation + Size
                    except Warning as e:
                        v_hat_stable/=v_hat_stable.max()
                        operation = operation + Size
                        
                        v_hat_stable[v_hat_stable<1e-150] = 1e-150
                        vstep = conbeta/v_hat_stable
                        operation = operation + Size
                        

                vstep/=vstep.max()
                operation = operation +Size
                
                xi[Size:]+=np.log(vstep)
                operation = operation + Size*2
                
                Z=B_stable*vstep[None,:]
                operation = operation + 2*Size*Size -Size
                
            f_eta_new=gamma*(np.log(Z.sum())+max_logB-xi[:Size].dot(conalpha)-xi[Size:].dot(conbeta))
            operation = operation+ Size*Size +2*Size+4
            
            operation = operation + 3
            if f_eta_new <= f_xi - (norm2_grad_f_xi)/2/L_new: # can be optimized 2 itmes
                primal_var = (alpha_new * B_stable/Bs_stable + L * alpha**2 * primal_var) /(L_new*alpha_new**2)
                operation = operation + 4*Size*Size + 5
                
                zeta -= alpha_new * grad_f_xi
                operation = operation + 4*Size
                
                eta = xi.copy()
                alpha = alpha_new
                L = L_new
                
                break
            L_new*=step
            operation = operation + 1
            
            


        valueot = np.dot(C.reshape(1,-1),primal_var.reshape(-1,1))[0][0]
        
        trueacc=abs(np.squeeze(primal_var.sum(axis=1).reshape(-1,1)) - conalpha).sum() + abs(np.squeeze(primal_var.sum(axis=0).reshape(-1, 1)) - conbeta).sum()
        Store_Acc.append(trueacc)
        Store_Oper.append(operation)

        Store_value.append(valueot)
        
          
        
        if  trueacc <= acc:        
            return primal_var, operation
   
        
def roundto(F, alpha, beta): #round to feasible area
    min_vec = np.vectorize(min)
    x = min_vec(alpha / F.sum(axis = 1).reshape(-1, 1),1)
    X = np.diag(x.T[0])
    F1 = np.dot(X, F)
    y = min_vec(beta / F1.sum(axis = 0).reshape(-1, 1),1)
    Y = np.diag(y.T[0])
    F11 = np.dot(F1, Y)
    erra = alpha - F11.sum(axis = 1).reshape(-1,1)
    errb = beta - F11.sum(axis = 0).reshape(-1,1)
    G = F11 + np.dot(erra, errb.T) / np.linalg.norm(erra, ord = 1)
    return G
       
def approxtoAAM(epsilon, Size, numberC,numberdistribution): 
    alpha = p_list_pre[numberdistribution]
    beta = q_list_pre[numberdistribution]
    C = C_store[numberC]
    penalty = epsilon / (4 * np.log(Size))
    epsilonprime = epsilon / 8 * abs(C).max()
    alphatilde = (1 - epsilonprime / 8) * alpha + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    betatilde = (1 - epsilonprime / 8) * beta + (epsilonprime / (8 * Size)) * np.repeat(1, Size).reshape(-1, 1)
    Xtilde, oper = AAM(acc = epsilonprime / 2, conalpha = np.squeeze(alphatilde), conbeta = np.squeeze(betatilde), C = C, Size = Size, gamma= penalty)
    Xhat = roundto(Xtilde, alpha = alpha, beta = beta)
    return Xhat, oper




#%%random choose 5 pairs of  MNIST images
p_list = [43, 31226,   239, 37372, 17390]
q_list = [45815, 35817, 43981, 4, 49947]


#%%a function to preprocess the MNIST data
def pre(mnistex):
    temp = np.array(mnistex,np.float64)
    index = np.where(temp == 0)[0]
    temp[index] = 1e-6
    temp/=temp.sum()
    temp=temp.reshape(-1,1)
    return temp


#%% resize the MNIST data
p_list_pre=[]
q_list_pre=[]

for i in range(5):
    array = np.array(images[p_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((3,3))
    testp=np.array(test).reshape(1,-1).tolist()[0]
    
    array = np.array(images[q_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((3,3))
    testq=np.array(test).reshape(1,-1).tolist()[0] 
    
    p_list_pre.append( pre( testp  ) )
    q_list_pre.append( pre( testq  ) )




for i in range(5):
    array = np.array(images[p_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((5,5))
    testp=np.array(test).reshape(1,-1).tolist()[0]
    
    array = np.array(images[q_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((5,5))
    testq=np.array(test).reshape(1,-1).tolist()[0] 
    
    p_list_pre.append( pre( testp  ) )
    q_list_pre.append( pre( testq  ) )


for i in range(5):
    array = np.array(images[p_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((7,7))
    testp=np.array(test).reshape(1,-1).tolist()[0]
    
    array = np.array(images[q_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((7,7))
    testq=np.array(test).reshape(1,-1).tolist()[0] 
    
    p_list_pre.append( pre( testp  ) )
    q_list_pre.append( pre( testq  ) )


for i in range(5):
    array = np.array(images[p_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((9,9))
    testp=np.array(test).reshape(1,-1).tolist()[0]
    
    array = np.array(images[q_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((9,9))
    testq=np.array(test).reshape(1,-1).tolist()[0] 
    
    p_list_pre.append( pre( testp  ) )
    q_list_pre.append( pre( testq  ) )
    
    
    
for i in range(5):
    array = np.array(images[p_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((10,10))
    testp=np.array(test).reshape(1,-1).tolist()[0]
    
    array = np.array(images[q_list[i]], dtype=np.uint8).reshape(28,28)

    # Use PIL to create an image from the new array of pixels
    new_image = Image.fromarray(array)
    test = new_image.resize((10,10))
    testq=np.array(test).reshape(1,-1).tolist()[0] 
    
    p_list_pre.append( pre( testp  ) )
    q_list_pre.append( pre( testq  ) )   

#%%Corresponding cost matrices
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)
C_store = []
mlist= [3,5,7,9,10]
for m in mlist:
    C = np.arange(m)
    C = cartesian_product(C, C)
    C = cdist(C, C, 'minkowski', p=1)
    C /= np.max(C)
    C_store.append(C)

###########
# Compare #
########### 
#%% Sinkhorn
def approxtoSinkhorn_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtoSinkhorn(epsilon = setting["epsilon"], Size = setting["Size"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])


settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49 ,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })    
print(settings)
print(cpu_count())
pool = Pool(12)
SinkhornResultsnsyn = pool.map(approxtoSinkhorn_wrap, [x for x in settings])
pool.close()
#%%PDASGD
def approxtoPDASGD_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtoPDASGD(epsilon = setting["epsilon"], Size = setting["Size"], inner_iters = setting["inner_iters"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])



settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "inner_iters": 6,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "inner_iters": 14,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49,
                   "inner_iters": 14,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "inner_iters": 18,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "inner_iters": 20,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })
    

print(settings)
print(cpu_count())
pool = Pool(12)
PDASGDResultsnsyn = pool.map(approxtoPDASGD_wrap, [x for x in settings])
pool.close()

#%%PDASMD
def approxtoPDASMD_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtoPDASMD(epsilon = setting["epsilon"], Size = setting["Size"], inner_iters = setting["inner_iters"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])



settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "inner_iters": 9,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "inner_iters": 25,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49,
                   "inner_iters": 49,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "inner_iters": 81,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "inner_iters": 100,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })
    

print(settings)
print(cpu_count())
pool = Pool(12)
PDASMDResultsnsyn = pool.map(approxtoPDASMD_wrap, [x for x in settings])
pool.close()


#%%AAM
def approxtoAAM_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtoAAM(epsilon = setting["epsilon"], Size = setting["Size"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])



settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })
    

print(settings)
print(cpu_count())
pool = Pool(12)
AAMResultsnsyn = pool.map(approxtoAAM_wrap, [x for x in settings])
pool.close()
#%%APDAGD
def approxtoAPDAGD_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtoAPDAGD(epsilon = setting["epsilon"], Size = setting["Size"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])



settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })
    

print(settings)
print(cpu_count())
pool = Pool(12)
APDAGDResultsnsyn= pool.map(approxtoAPDAGD_wrap, [x for x in settings])
pool.close()

#%%stochastic Sinkhorn
def approxtostochasticsinkhorn_wrap(setting):
     created = multiprocess.Process()
     current = multiprocess.current_process()
     print ('running:', current.name, current._identity)
     print ('created:', created.name, created._identity)
     return approxtostochasticsinkhorn(epsilon = setting["epsilon"], Size = setting["Size"],numberC = setting["numberC"], numberdistribution = setting["numberdistribution"])



settings = []
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 9,
                   "numberC": 0,
                   "numberdistribution": i,
                   })
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 25,
                   "numberC": 1,
                   "numberdistribution": 5+i,
                   })    

    
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 49,
                   "numberC": 2,
                   "numberdistribution": 10+i,
                   }) 
for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 81,
                   "numberC": 3,
                   "numberdistribution": 15+i,
                   })     

for i in range(5):
    settings.append({
                   "epsilon": 0.02,
                   "Size": 100,
                   "numberC": 4,
                   "numberdistribution": 20+i,
                   })
    

print(settings)
print(cpu_count())
pool = Pool(12)
stochasticsinkhornResultsstoch5syn = pool.map(approxtostochasticsinkhorn_wrap, [x for x in settings])
pool.close()

#%%
APDAGD5 = [APDAGDResultsnsyn[0][1], APDAGDResultsnsyn[1][1], APDAGDResultsnsyn[2][1], APDAGDResultsnsyn[3][1], APDAGDResultsnsyn[4][1]]
APDAGD7 = [APDAGDResultsnsyn[5][1], APDAGDResultsnsyn[6][1], APDAGDResultsnsyn[7][1], APDAGDResultsnsyn[8][1], APDAGDResultsnsyn[9][1]]
APDAGD10 = [APDAGDResultsnsyn[10][1], APDAGDResultsnsyn[11][1], APDAGDResultsnsyn[12][1], APDAGDResultsnsyn[13][1], APDAGDResultsnsyn[14][1]]
APDAGD15 = [APDAGDResultsnsyn[15][1], APDAGDResultsnsyn[16][1], APDAGDResultsnsyn[17][1], APDAGDResultsnsyn[18][1], APDAGDResultsnsyn[19][1]]
APDAGD20 = [APDAGDResultsnsyn[20][1], APDAGDResultsnsyn[21][1], APDAGDResultsnsyn[22][1], APDAGDResultsnsyn[23][1], APDAGDResultsnsyn[24][1]]

APDAGDmean = np.array([statistics. mean(APDAGD5),statistics. mean(APDAGD7),statistics. mean(APDAGD10),statistics. mean(APDAGD15),statistics. mean(APDAGD20)])
APDAGDmean = np.log(APDAGDmean)

APDAGDstd =  np.array([np.log(np.array(APDAGD5)).std(),np.log(np.array(APDAGD7)).std(),np.log(np.array(APDAGD10)).std(),np.log(np.array(APDAGD15)).std(),np.log(np.array(APDAGD20)).std()])




#%%
AAM5 = [AAMResultsnsyn[0][1], AAMResultsnsyn[1][1], AAMResultsnsyn[2][1], AAMResultsnsyn[3][1], AAMResultsnsyn[4][1]]
AAM7 = [AAMResultsnsyn[5][1], AAMResultsnsyn[6][1], AAMResultsnsyn[7][1], AAMResultsnsyn[8][1], AAMResultsnsyn[9][1]]
AAM10 = [AAMResultsnsyn[10][1], AAMResultsnsyn[11][1], AAMResultsnsyn[12][1], AAMResultsnsyn[13][1], AAMResultsnsyn[14][1]]
AAM15 = [AAMResultsnsyn[15][1], AAMResultsnsyn[16][1], AAMResultsnsyn[17][1], AAMResultsnsyn[18][1], AAMResultsnsyn[19][1]]
AAM20 = [AAMResultsnsyn[20][1], AAMResultsnsyn[21][1], AAMResultsnsyn[22][1], AAMResultsnsyn[23][1], AAMResultsnsyn[24][1]]

AAMmean = np.array([statistics. mean(AAM5),statistics. mean(AAM7),statistics. mean(AAM10),statistics. mean(AAM15),statistics. mean(AAM20)])
AAMmean = np.log(AAMmean)

AAMstd =  np.array([np.log(np.array(AAM5)).std(),np.log(np.array(AAM7)).std(),np.log(np.array(AAM10)).std(),np.log(np.array(AAM15)).std(),np.log(np.array(AAM20)).std()])




#%%
Sinkhorn5 = [SinkhornResultsnsyn[0][1], SinkhornResultsnsyn[1][1], SinkhornResultsnsyn[2][1], SinkhornResultsnsyn[3][1], SinkhornResultsnsyn[4][1]]
Sinkhorn7 = [SinkhornResultsnsyn[5][1], SinkhornResultsnsyn[6][1], SinkhornResultsnsyn[7][1], SinkhornResultsnsyn[8][1], SinkhornResultsnsyn[9][1]]
Sinkhorn10 = [SinkhornResultsnsyn[10][1], SinkhornResultsnsyn[11][1], SinkhornResultsnsyn[12][1], SinkhornResultsnsyn[13][1], SinkhornResultsnsyn[14][1]]
Sinkhorn15 = [SinkhornResultsnsyn[15][1], SinkhornResultsnsyn[16][1], SinkhornResultsnsyn[17][1], SinkhornResultsnsyn[18][1], SinkhornResultsnsyn[19][1]]
Sinkhorn20 = [SinkhornResultsnsyn[20][1], SinkhornResultsnsyn[21][1], SinkhornResultsnsyn[22][1], SinkhornResultsnsyn[23][1], SinkhornResultsnsyn[24][1]]

Sinkhornmean = np.array([statistics. mean(Sinkhorn5),statistics. mean(Sinkhorn7),statistics. mean(Sinkhorn10),statistics. mean(Sinkhorn15),statistics. mean(Sinkhorn20)])
Sinkhornmean = np.log(Sinkhornmean)
Sinkhornstd =  np.array([np.log(np.array(Sinkhorn5)).std(),np.log(np.array(Sinkhorn7)).std(),np.log(np.array(Sinkhorn10)).std(),np.log(np.array(Sinkhorn15)).std(),np.log(np.array(Sinkhorn20)).std()])
#Sinkhornstd = np.log(Sinkhornstd)


#%%
PDASGD5 = [PDASGDResultsnsyn[0][1], PDASGDResultsnsyn[1][1], PDASGDResultsnsyn[2][1], PDASGDResultsnsyn[3][1], PDASGDResultsnsyn[4][1]]
PDASGD7 = [PDASGDResultsnsyn[5][1], PDASGDResultsnsyn[6][1], PDASGDResultsnsyn[7][1], PDASGDResultsnsyn[8][1], PDASGDResultsnsyn[9][1]]
PDASGD10 = [PDASGDResultsnsyn[10][1], PDASGDResultsnsyn[11][1], PDASGDResultsnsyn[12][1], PDASGDResultsnsyn[13][1], PDASGDResultsnsyn[14][1]]
PDASGD15 = [PDASGDResultsnsyn[15][1], PDASGDResultsnsyn[16][1], PDASGDResultsnsyn[17][1], PDASGDResultsnsyn[18][1], PDASGDResultsnsyn[19][1]]
PDASGD20 = [PDASGDResultsnsyn[20][1], PDASGDResultsnsyn[21][1], PDASGDResultsnsyn[22][1], PDASGDResultsnsyn[23][1], PDASGDResultsnsyn[24][1]]

PDASGDmean = np.array([statistics. mean(PDASGD5),statistics. mean(PDASGD7),statistics. mean(PDASGD10),statistics. mean(PDASGD15),statistics. mean(PDASGD20)])
PDASGDmean = np.log(PDASGDmean)


PDASGDstd =  np.array([np.log(np.array(PDASGD5)).std(),np.log(np.array(PDASGD7)).std(),np.log(np.array(PDASGD10)).std(),np.log(np.array(PDASGD15)).std(),np.log(np.array(PDASGD20)).std()])

#%%PDASMD
PDASMD5 = [PDASMDResultsnsyn[0][1], PDASMDResultsnsyn[1][1], PDASMDResultsnsyn[2][1], PDASMDResultsnsyn[3][1], PDASMDResultsnsyn[4][1]]
PDASMD7 = [PDASMDResultsnsyn[5][1], PDASMDResultsnsyn[6][1], PDASMDResultsnsyn[7][1], PDASMDResultsnsyn[8][1], PDASMDResultsnsyn[9][1]]
PDASMD10 = [PDASMDResultsnsyn[10][1], PDASMDResultsnsyn[11][1], PDASMDResultsnsyn[12][1], PDASMDResultsnsyn[13][1], PDASMDResultsnsyn[14][1]]
PDASMD15 = [PDASMDResultsnsyn[15][1], PDASMDResultsnsyn[16][1], PDASMDResultsnsyn[17][1], PDASMDResultsnsyn[18][1], PDASMDResultsnsyn[19][1]]
PDASMD20 = [PDASMDResultsnsyn[20][1], PDASMDResultsnsyn[21][1], PDASMDResultsnsyn[22][1], PDASMDResultsnsyn[23][1], PDASMDResultsnsyn[24][1]]

PDASMDmean = np.array([statistics. mean(PDASMD5),statistics. mean(PDASMD7),statistics. mean(PDASMD10),statistics. mean(PDASMD15),statistics. mean(PDASMD20)])
PDASMDmean = np.log(PDASMDmean)

PDASMDstd =  np.array([np.log(np.array(PDASMD5)).std(),np.log(np.array(PDASMD7)).std(),np.log(np.array(PDASMD10)).std(),np.log(np.array(PDASMD15)).std(),np.log(np.array(PDASMD20)).std()])

#%%Stochastic sinkhorn
stochasticsinkhornDatastoch5syn = stochasticsinkhornResultsstoch5syn
stochasticsinkhorn2 = [stochasticsinkhornDatastoch5syn[0][1], stochasticsinkhornDatastoch5syn[1][1], stochasticsinkhornDatastoch5syn[2][1], stochasticsinkhornDatastoch5syn[3][1], stochasticsinkhornDatastoch5syn[4][1]]
stochasticsinkhorn4 = [stochasticsinkhornDatastoch5syn[5][1], stochasticsinkhornDatastoch5syn[6][1], stochasticsinkhornDatastoch5syn[7][1], stochasticsinkhornDatastoch5syn[8][1], stochasticsinkhornDatastoch5syn[9][1]]
stochasticsinkhorn6 = [stochasticsinkhornDatastoch5syn[10][1], stochasticsinkhornDatastoch5syn[11][1], stochasticsinkhornDatastoch5syn[12][1], stochasticsinkhornDatastoch5syn[13][1], stochasticsinkhornDatastoch5syn[14][1]]
stochasticsinkhorn8 = [stochasticsinkhornDatastoch5syn[15][1], stochasticsinkhornDatastoch5syn[16][1], stochasticsinkhornDatastoch5syn[17][1], stochasticsinkhornDatastoch5syn[18][1], stochasticsinkhornDatastoch5syn[19][1]]
stochasticsinkhorn10 = [stochasticsinkhornDatastoch5syn[20][1], stochasticsinkhornDatastoch5syn[21][1], stochasticsinkhornDatastoch5syn[22][1], stochasticsinkhornDatastoch5syn[23][1], stochasticsinkhornDatastoch5syn[24][1]]

stochasticsinkhornmean = np.array([statistics. mean(stochasticsinkhorn2),statistics. mean(stochasticsinkhorn4),statistics. mean(stochasticsinkhorn6),statistics. mean(stochasticsinkhorn8),statistics. mean(stochasticsinkhorn10)])
stochasticsinkhornmean = np.log(stochasticsinkhornmean)

stochasticsinkhornstd =  np.array([np.log(np.array(stochasticsinkhorn2)).std(),np.log(np.array(stochasticsinkhorn4)).std(),np.log(np.array(stochasticsinkhorn6)).std(),np.log(np.array(stochasticsinkhorn8)).std(),np.log(np.array(stochasticsinkhorn10)).std()])




#%%plot the results
n = np.array([9,25,49,81,100])
logn = np.log(n)

plt.figure(dpi=1000)
import matplotlib.pyplot as plt
plt.plot(logn, Sinkhornmean,label="Sinkhorn",color='mediumvioletred', marker ='o',markersize = 5,lw=1)
plt.fill_between(logn,Sinkhornmean - Sinkhornstd,Sinkhornmean + Sinkhornstd,color='mediumvioletred',alpha=0.2)



plt.plot(logn, AAMmean, label="AAM",color='black', marker ='v',markersize = 5,lw=1)
plt.fill_between(logn,AAMmean - AAMstd,AAMmean + AAMstd,color='black',alpha=0.2)

plt.plot(logn, APDAGDmean, label = "APDAGD",color='darkkhaki',marker ='^',markersize = 5,lw=1)
plt.fill_between(logn,APDAGDmean - APDAGDstd,APDAGDmean + APDAGDstd,color='darkkhaki',alpha=0.2)

plt.plot(logn, PDASGDmean,label = "PDASGD",color='blue',marker ='x',markersize = 5,lw=1)
plt.fill_between(logn,PDASGDmean - PDASGDstd,PDASGDmean + PDASGDstd,color='blue',alpha=0.2)

plt.plot(logn, PDASMDmean,label = "PDASMD",color='crimson',marker ='>',markersize = 5,lw=1)
plt.fill_between(logn,PDASMDmean - PDASMDstd,PDASMDmean + PDASMDstd,color='crimson',alpha=0.2)

plt.plot(logn, stochasticsinkhornmean, label = "stochastic Sinkhorn",color='green',marker = 'D',markersize = 5,lw=1)
plt.fill_between(logn,stochasticsinkhornmean - stochasticsinkhornstd,stochasticsinkhornmean + stochasticsinkhornstd,color='green',alpha=0.2)

plt.xlabel(r'$ln(n)$',fontsize=20)
plt.ylabel(r'$ln( \# operation )$',fontsize=20) 
plt.legend(fontsize=10)
plt.title("ln(number of operations)",fontsize=20)
plt.ylim(13,24)


plt.savefig("MNISTcompare.pdf", format="pdf")






