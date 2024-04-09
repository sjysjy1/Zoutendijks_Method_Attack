import torch
import torch.nn as nn
from torch import Tensor
def loss(model,images,labels):
    #optimizer.zero_grad()
    outputs = model(images)
    CELoss = nn.CrossEntropyLoss(reduction='none')
    loss = CELoss(outputs, labels)
    _, predictions_labels = torch.max(outputs, 1)
    return loss, predictions_labels



def zoutendijk_attack_Linfty_M1(model:nn.Module,
                             images:Tensor,
                             labels:Tensor,
                             epsilon:float=0.3,
                             iter_max: int = 20,
                             stepsize_default:float=0.01,
                             early_stop:bool=False,
                             beta: float=0.3,
                             rho:float=0.999,
                             div=1.0
                             ) -> Tensor:
    '''
    :param model: model to attack
    :param images: (batch,channel,height,weight)
    :param labels: correct label
    :param epsilon: perturbation budget
    :param iter_max: maximum number of iteration
    :param stepsize_default: initial step size
    :param beta: decaying rate for momentum
    :param rho: decaying rate for RMSProp algorithm
    :param div: division parameter for maximum step size
    :return:
    '''
    iter=0
    device=images.device
    adv_best = images.clone().detach().to(device)
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    one_tensor=torch.ones(images.shape, dtype=torch.float32, device=device)
    zero_tensor = torch.zeros(images.shape, dtype=torch.float32, device=device)
    upper_bound = torch.min(images + epsilon,one_tensor)
    lower_bound = torch.max(images - epsilon,zero_tensor)
    loss_best=torch.full((len(images), ),-torch.inf,device=device)
    success=torch.full((len(images), ),False,device=device)
    momentum = torch.zeros_like(images).to(device)
    RMS_avg=torch.zeros_like(images).to(device)

    images.requires_grad_()
    model.zero_grad()
    loss_val,predictions_attacked = loss(model, images, labels)
    (-loss_val.sum()).backward()
    while iter<iter_max:
        iter+=1

        #incorporate momentum
        direction = torch.zeros(images.shape, dtype=torch.float, device=device)
        grad_sum = images.grad / torch.norm((images.grad).view(len(images), -1), p=1, dim=1)[:, None, None, None]
        grad_sum = grad_sum + beta * momentum
        momentum = grad_sum
        direction[(grad_sum>=0)]=-1
        direction[(grad_sum < 0)] = 1

        #using RMSProp to determine step size
        inf_mat1=torch.full_like(images, 1.0e20, device=device)
        inf_mat2=torch.full_like(images, 1.0e20, device=device)
        inf_mat1[(direction==-1)]=images[(direction==-1)]
        inf_mat2[(direction==1)]=images[(direction==1)]
        stepsize_max=torch.min(inf_mat1-lower_bound,torch.abs(upper_bound-inf_mat2))
        RMS_avg = rho * RMS_avg + (1 - rho) * direction ** 2
        stepsize = torch.full_like(stepsize_max, stepsize_default, device=device)
        stepsize = torch.min(stepsize/(RMS_avg**0.5+0.000000001),stepsize_max/div)

        #update image
        if early_stop:
           with torch.no_grad():
              images[~success] = (images[~success] + (stepsize * direction)[~success])
           images.requires_grad_()
           model.zero_grad()
           loss_val, predictions_attacked = loss(model, images, labels)
           (-loss_val.sum()).backward()
           success = (predictions_attacked != labels)
           if torch.all(success):
               break
        else:
           images = (images + stepsize * direction).detach()
           images.requires_grad_()
           model.zero_grad()
           loss_val, predictions_attacked = loss(model, images, labels)
           (-loss_val.sum()).backward()
           is_better_adv = loss_val > loss_best
           loss_best=torch.where(is_better_adv, loss_val, loss_best)
           adv_best=torch.where(is_better_adv[:,None,None,None], images.detach(), adv_best)
    if early_stop:
        return  images
    else:
        return  adv_best




def zoutendijk_attack_Linfty_M2(model:nn.Module,
                             images:Tensor,
                             labels:Tensor,
                             epsilon:float=0.2,
                             stepsize_default:float=0.01,
                         #    delta:float=0.015,
                             iter_max:int=20,
                             div:float=2.0,
                             beta:float=0.3,
                             rho = 0.99,
                             early_stop:bool=False,
                             )-> Tensor:
    '''
    :param model: model to attack
    :param images: (batch,channel,height,width)
    :param labels: correct label
    :param epsilon: perturbation budget
    :param iter_max: maximum number of iteration
    :param stepsize_default: initial step size
    :param beta: decaying rate for momentum
    :param rho: decaying rate for RMSProp algorithm
    :param div: division parameter for maximum step size
    :return:
    '''
    iter=0
    device = images.device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    one_tensor=torch.ones(images.shape, dtype=torch.float32, device=device)
    zero_tensor = torch.zeros(images.shape, dtype=torch.float32, device=device)
    upper_bound = torch.min(images + epsilon,one_tensor)
    lower_bound = torch.max(images - epsilon,zero_tensor)
    loss_best=torch.full((len(images), ),-1,device=device)
    adv_best = images.clone().detach().to(device)
    success = torch.full((len(images),), False, device=device)
    momentum = torch.zeros_like(images).to(device)
    RMS_avg=torch.zeros_like(images).to(device)

    images.requires_grad_()
    model.zero_grad()
    loss_val,predictions_attacked = loss(model, images, labels)
    (-loss_val.sum()).backward()
    while iter<iter_max:
        iter+=1

        #incorporate momentum
        grad_norm=torch.norm((images.grad).view(len(images), -1), p=2, dim=1)
        grad_sum = images.grad / grad_norm[:, None, None, None]
        grad_sum = grad_sum + beta*momentum
        momentum = grad_sum
        direction = -grad_sum / (torch.norm((grad_sum).view(len(images), -1), p=2, dim=1) ** 0.5)[:, None, None, None]

        #using RMSProp to determing step size
        stepsize_max=torch.full_like(images,1.0e15,device=device)
        bound1=(images - lower_bound) / torch.abs(-direction)
        bound2=(upper_bound - images) / torch.abs(direction)
        bound1[~(grad_sum > 0)]=torch.inf
        bound2[~(grad_sum < 0)] = torch.inf
        stepsize_max =torch.min(stepsize_max,bound1)
        stepsize_max = torch.min(stepsize_max, bound2)
        RMS_avg = rho * RMS_avg + (1 - rho) * direction ** 2
        stepsize = torch.full_like(images, stepsize_default, device=device)
        stepsize = torch.min(stepsize/(RMS_avg**0.5+0.000000001),stepsize_max/div)

        #update image
        if early_stop:
            with torch.no_grad():
                 images[~success] = (images[~success] + (stepsize * direction)[~success])
            loss_val, predictions_attacked = loss(model, images, labels)
            (-loss_val.sum()).backward()
            success = (predictions_attacked != labels)
            if torch.all(success):
                break
        else:
            images = (images + stepsize * direction).detach()
            images.requires_grad_()
            loss_val, predictions_attacked = loss(model, images, labels)
            (-loss_val.sum()).backward()
            is_better_adv = loss_val > loss_best
            loss_best=torch.where(is_better_adv, loss_val, loss_best)
            adv_best=torch.where(is_better_adv[:,None,None,None], images.detach(), adv_best)
    if early_stop:
        return   images
    else:
        return  adv_best






