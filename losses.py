
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 7

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values 
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)
    
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight) 
    

def softmax_hard_mining_loss1(input_values,  delta):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    eps=1e-10
    p=torch.clip(p,eps,1.-eps)
    loss = ((torch.sin((p)*np.pi)/((p)*np.pi))-( (torch.exp(-delta*p))*(torch.sin((p)*delta*np.pi)/((p)*delta*np.pi))) )  * input_values 
    return loss.mean()

class SoftmaxHardMiningLoss1(nn.Module):
    def __init__(self, weight=None,delta=10000000):
        super(SoftmaxHardMiningLoss1, self).__init__()
        assert delta >= 0
        self.weight = weight
        self.delta=delta


    def forward(self, input, target):
        return softmax_hard_mining_loss1(F.cross_entropy(input, target, reduction='none', weight=self.weight),self.delta)
    

def softmax_hard_mining_loss2(input_values):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    eps=1e-10
    p=torch.clip(p,eps,1.-eps)
    loss = (torch.sin((p)*np.pi)/((p)*np.pi))  * input_values 
    return loss.mean()

class SoftmaxHardMiningLoss2(nn.Module):
    def __init__(self, weight=None):
        super(SoftmaxHardMiningLoss2, self).__init__()
        self.weight = weight


    def forward(self, input, target):
        return softmax_hard_mining_loss2(F.cross_entropy(input, target, reduction='none', weight=self.weight))


def softmax_hard_mining_loss1S10(input_values,  delta):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    eps=1e-10
    p=torch.clip(p,eps,1.-eps)
    loss = ((torch.sin((p)*np.pi)/((p)*np.pi))-( (torch.exp(-delta*p))*(torch.sin((p)*delta*np.pi)/((p)*delta*np.pi))) )  * input_values
    return loss.mean()

class SoftmaxHardMiningLoss1S10(nn.Module):
    def __init__(self, weight=None,delta=10000000000):
        super(SoftmaxHardMiningLoss1S10, self).__init__()
        assert delta >= 0
        self.weight = weight
        self.delta=delta


    def forward(self, input, target):
        return softmax_hard_mining_loss1(F.cross_entropy(input, target, reduction='none', weight=self.weight),self.delta)
class DynamicalMiningLoss2(nn.Module):
    def __init__(self, weight=None,max_thresh=0.3,epoch=0,numb_epochs=100):
        super(DynamicalMiningLoss2, self).__init__()
        self.weight = weight
        self.thresholds_list=max_thresh
        self.numb_epochs=numb_epochs
        self.threshold=0
        self.max_thresh=max_thresh
        self.current_epoch = epoch
        if weight==None:
              self.weight=torch.tensor(1.)
        else:
              self.weight=torch.tensor(weight)

    def forward(self, input, target):
        y_pred=F.softmax(input,dim=1)
        eps=1e-10
        y_pred=torch.clip(y_pred,eps,1.-eps)
        self.threshold=(self.max_thresh/self.numb_epochs)*self.current_epoch
        is_negative=y_pred<=self.threshold
        y_pred_new=torch.where(is_negative,1-y_pred,y_pred-self.threshold)
        y_true=F.one_hot(target,num_classes=num_classes)
        pt=((torch.sin((y_pred_new)*np.pi)/((y_pred_new)*np.pi))) 
        dml=-self.weight*pt*torch.log(y_pred_new)        
        #factor=torch.sum(pt*y_true,1)
        # cross_entropy=y_true*torch.log(y_pred)
        # loss=-cross_entropy*self.weight
        loss=torch.sum(y_true*dml,dim=1)
        return loss.mean()
class DynamicalMiningLoss1(nn.Module):
    def __init__(self, weight=None,max_thresh=0.3,epoch=0,numb_epochs=100,delta=10000000):
        super(DynamicalMiningLoss1, self).__init__()
        self.weight = weight
        self.thresholds_list=max_thresh
        self.numb_epochs=numb_epochs
        self.threshold=0
        self.max_thresh=max_thresh
        self.current_epoch = epoch
        self.delta=delta
        if weight==None:
              self.weight=torch.tensor(1.)
        else:
              self.weight=torch.tensor(weight)

    def forward(self, input, target):
        y_pred=F.softmax(input,dim=1)
        eps=1e-10
        y_pred=torch.clip(y_pred,eps,1.-eps)
        self.threshold=(self.max_thresh/self.numb_epochs)*self.current_epoch
        is_negative=y_pred<=self.threshold
        y_pred_new=torch.where(is_negative,1-y_pred,y_pred-self.threshold)
        y_true=F.one_hot(target,num_classes=num_classes)
        pt=((torch.sin((y_pred_new)*np.pi)/((y_pred_new)*np.pi))-( (torch.exp(-self.delta*y_pred_new))*(torch.sin((y_pred_new)*self.delta*np.pi)/((y_pred_new)*self.delta*np.pi))) )
        dml=-self.weight*pt*torch.log(y_pred_new)        
        #factor=torch.sum(pt*y_true,1)
        # cross_entropy=y_true*torch.log(y_pred)
        # loss=-cross_entropy*self.weight
        loss=torch.sum(y_true*dml,dim=1)
        return loss.mean()
