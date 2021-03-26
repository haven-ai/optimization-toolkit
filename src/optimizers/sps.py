
import torch
import copy
import time
import numpy as np

from src.optimizers.base import utils as ut

MOM_FLAGS = ['mom1','mom2', 'mom3', 'mom3_old', 'mom2_old']
class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 eta_max=10,
                 adapt_flag=None,
                 fstar_flag=None,
                 eps=0,
                 momentum=0, 
                 mom_flag=None):
        params = list(params)
        self.mom_flag = mom_flag
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0
        self.state['step_size_avg'] = 0.
        self.momentum = momentum

        self.params_prev = None
        if self.adapt_flag in MOM_FLAGS:
            self.params_prev = copy.deepcopy(params) 

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['lb'] = None
        self.loss_min = np.inf
        self.loss_sum = 0.
        self.loss_max = 0.
        self.fstar_flag = fstar_flag

    def step(self, closure, clip_grad=False):
        # deterministic closure
        seed = time.time()

        batch_step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        if self.state['step'] % int(self.n_batches_per_epoch) == 0:
            self.state['step_size_avg'] = 0.

        self.state['step'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)

        grad_norm = ut.compute_grad_norm(grad_current)
        loss_curr = loss.item()

        if self.adapt_flag in ['smooth_iter']:
            step_size = loss / (self.c * (grad_norm)**2)
            coeff = self.gamma**(1./self.n_batches_per_epoch)
            step_size =  min(coeff * self.state['step_size'], step_size.item())

        elif self.adapt_flag in ['mom1']:
            step_size = loss / (self.c * (grad_norm)**2)
            coeff = self.gamma**(1./self.n_batches_per_epoch)
            step_size =  min(coeff * self.state['step_size'], 
                             step_size.item())
            beta = self.momentum 
            loss_curr = loss.item()

        elif self.adapt_flag in ['mom4', 'mom2', 'mom3', 'mom2_smooth']:
            loss_prev = f_xk_old = self.state.get('loss_prev', 0)
            f_xk = loss_curr
            
            # get beta
            # --------
            if self.adapt_flag in ['mom2_smooth', 'mom2']:
                if loss_curr < loss_prev:
                    # if loss is increasing
                    beta = 0
                else:
                    beta = self.momentum

            elif self.adapt_flag == 'mom3':
                if loss_curr < loss_prev:
                    # if loss is increasing
                    beta = 0
                else:
                    lhs = loss_curr / (loss_prev - loss_curr)
                    # print((1 + lhs)*loss_curr - lhs * loss_prev)
                    beta = self.momentum

            elif self.adapt_flag == 'mom4':
                lhs = f_xk / max(f_xk_old, 1e-8)
                # print(f_xk, f_xk_old)
                beta = self.momentum
                while  lhs <= beta:
                    assert(lhs < 1)
                    beta *= 0.9
            
            # get step size
            # -------------
            if self.adapt_flag == 'mom4':
                step_size = f_xk - beta * f_xk_old
                step_size /= (self.c * (grad_norm)**2)

            elif self.adapt_flag in ['mom2', 'mom3']:
                step_size = (1 + beta)*loss_curr - beta * loss_prev
                step_size /= (self.c * (grad_norm)**2)

            elif self.adapt_flag == 'mom2_smooth':
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                step_size =  min(coeff * self.state['step_size'], 
                                step_size.item())
       
                

            self.state['loss_prev'] = f_xk
        
        # update with step size
        if self.adapt_flag in MOM_FLAGS:
            params_tmp = copy.deepcopy(self.params) 
            for p, g, p_prev in zip(self.params, grad_current, self.params_prev):
                p.data = p - step_size * g + beta * (p - p_prev)
            self.params_prev = params_tmp
        else:
            for p, g in zip(self.params, grad_current):
                p.data.add_(alpha=- float(step_size), tensor=g)
            
        # save the new step-size
        self.state['step_size'] = float(step_size)

        
        self.state['step_size_avg'] += (step_size / self.n_batches_per_epoch)
        self.state['grad_norm'] = grad_norm.item()
        
        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected.')
        
        return loss
