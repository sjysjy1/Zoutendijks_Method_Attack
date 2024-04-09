import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_MNIST import model_MNIST

from zoutendijk_attack import zoutendijk_attack_Linfty_M1, zoutendijk_attack_Linfty_M2
import time
import matplotlib.pyplot as plt
import torchattacks
import random

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
def loss(model,images,labels):
    #optimizer.zero_grad()
    outputs = model(images)
    CELoss = nn.CrossEntropyLoss(reduction='none')
    loss = CELoss(outputs, labels)
    _, predictions_labels = torch.max(outputs, 1)
    return loss, predictions_labels

seed_torch()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())

model=model_MNIST()
model.to(device)
criterion=nn.CrossEntropyLoss()
batch_size=256
list_para=[
{'model':'Standard','attack':'Z_linf_M1','epsilon':0.2,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M1','epsilon':0.3,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M1','epsilon':0.4,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':0.2,'stepsize':0.01,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':0.3,'stepsize':0.01,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':0.4,'stepsize':0.01,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'Standard','attack':'PGD','epsilon':0.2,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'PGD','epsilon':0.3,'stepsize':0.015,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'PGD','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'BIM','epsilon':0.2,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'BIM','epsilon':0.3,'stepsize':0.015,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'BIM','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':0.2,'stepsize':0.01,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':0.3,'stepsize':0.015,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':0.4,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'NIFGSM','epsilon':0.2,'stepsize':0.01,'iter_max':20, 'batch_size':batch_size,'decay':0.3},
{'model':'Standard','attack':'NIFGSM','epsilon':0.3,'stepsize':0.015,'iter_max':20,'batch_size':batch_size,'decay':0.3},
{'model':'Standard','attack':'NIFGSM','epsilon':0.4,'stepsize':0.02,'iter_max':20, 'batch_size':batch_size,'decay':0.3},


{'model':'ddn','attack':'Z_linf_M1','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'ddn','attack':'Z_linf_M1','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'ddn','attack':'Z_linf_M1','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'ddn','attack':'Z_linf_M2','epsilon':0.2,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'ddn','attack':'Z_linf_M2','epsilon':0.3,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'ddn','attack':'Z_linf_M2','epsilon':0.4,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'ddn','attack':'PGD','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'ddn','attack':'PGD','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'ddn','attack':'PGD','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'ddn','attack':'BIM','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'BIM','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'BIM','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'MIFGSM','epsilon':0.2,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'MIFGSM','epsilon':0.3,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'MIFGSM','epsilon':0.4,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'ddn','attack':'NIFGSM','epsilon':0.2,'stepsize':0.02,'iter_max':20, 'batch_size':batch_size,'decay':0.3},
{'model':'ddn','attack':'NIFGSM','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'decay':0.3},
{'model':'ddn','attack':'NIFGSM','epsilon':0.4,'stepsize':0.02,'iter_max':20, 'batch_size':batch_size,'decay':0.3},


{'model':'trades','attack':'Z_linf_M1','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'trades','attack':'Z_linf_M1','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'trades','attack':'Z_linf_M1','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'div':1.0,'early_stop':False},
{'model':'trades','attack':'Z_linf_M2','epsilon':0.2,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'trades','attack':'Z_linf_M2','epsilon':0.3,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'trades','attack':'Z_linf_M2','epsilon':0.4,'stepsize':0.02,'iter_max':20,'div':1.0,'batch_size':batch_size,'beta':0.3,'rho': 0.999,'early_stop':False},
{'model':'trades','attack':'PGD','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'trades','attack':'PGD','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'trades','attack':'PGD','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'trades','attack':'BIM','epsilon':0.2,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'BIM','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'BIM','epsilon':0.4,'stepsize':0.02,'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'MIFGSM','epsilon':0.2,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'MIFGSM','epsilon':0.3,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'MIFGSM','epsilon':0.4,'stepsize':0.02,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'trades','attack':'NIFGSM','epsilon':0.2,'stepsize':0.02,'iter_max':20, 'batch_size':batch_size,'decay':0.3},
{'model':'trades','attack':'NIFGSM','epsilon':0.3,'stepsize':0.02,'iter_max':20,'batch_size':batch_size,'decay':0.3},
{'model':'trades','attack':'NIFGSM','epsilon':0.4,'stepsize':0.02,'iter_max':20, 'batch_size':batch_size,'decay':0.3},

]

for item in list_para:
    list_success_fail=[]
    list_pert_linf=[]
    list_pert_l2 = []
    list_pert_l1 = []
    list_diff=[]
    list_num=[]
    list_loss=[]
    start_time = time.time()
    print(item)

    model = model_MNIST()
    if item['model'] == 'Standard':
        model.load_state_dict(torch.load('./models/mnist/mnist_regular.pth'), False)
    elif item['model'] == 'ddn':
        model.load_state_dict(torch.load('./models/mnist/mnist_robust_ddn.pth'), False)
    elif item['model'] == 'trades':
        model.load_state_dict(torch.load('./models/mnist/mnist_robust_trades.pt'),False)
    model.to(device)
    model.eval()  # turn off the dropout
    test_data = torch.unsqueeze(test_dataset.data, dim=1)
    test_labels = test_dataset.test_labels.to(device)
    test_data_normalized = test_data / 255.0
    test_data_normalized = test_data_normalized.to(device)
    outputs = model(test_data_normalized)
    _, labels_predict = torch.max(outputs, 1)
    correct = torch.eq(labels_predict, test_labels)
    correct_sum = correct.sum()
    correct_index=[]
    for i in range(10000):
        if correct[i]:
            correct_index.append(i)
    #print(correct.sum())
    print('clean accuracy is:', correct_sum / 10000.0)
    start_time = time.time()
    if item['attack']=='Z_linf_M1':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]


            adv_images=zoutendijk_attack_Linfty_M1(model,images,labels,epsilon=item['epsilon'],iter_max=item['iter_max'],stepsize_default=item['stepsize'],beta=item['beta'],rho=item['rho'],div=item['div'])

            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1)
            list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
            list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
            with torch.no_grad():
                loss_val, labels_predict = loss(model, adv_images, labels)
            success=(labels_predict!=labels)
            list_success_fail = list_success_fail + success.tolist()
            loss_avg = sum(loss_val) / len(loss_val)
            print('perturbation is: ', torch.t(pert_linf))
            print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
            # list_num.append(iter)
            list_loss += loss_val.tolist()

    elif item['attack'] == 'Z_linf_M2':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]

            adv_images = zoutendijk_attack_Linfty_M2(model, images, labels,epsilon=item['epsilon'],iter_max=item['iter_max'],stepsize_default=item['stepsize'],beta=item['beta'],rho=item['rho'],div=item['div'])

            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1)
            list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
            list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
            with torch.no_grad():
                loss_val, labels_predict = loss(model, adv_images, labels)
            success=(labels_predict!=labels)
            list_success_fail = list_success_fail + success.tolist()
            loss_avg = sum(loss_val) / len(loss_val)
            print('perturbation is: ', torch.t(pert_linf))
            print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
            # list_num.append(iter)
            list_loss += loss_val.tolist()

    elif item['attack']=='PGD':
           for i in range(0, len(correct_index), item['batch_size']):
               print('***************{}th batch***********'.format(int(i / item['batch_size'])))
               images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
               labels = test_labels[correct_index[i:i + item['batch_size']]]

               atk = torchattacks.PGD(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'],random_start=True)  # torchattack
               adv_images, perturbation, _, success, _, predict_label = atk(images, labels)
               pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1), dim=1)
               pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
               pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

               list_success_fail = list_success_fail + success.tolist()
               list_pert_linf +=  torch.squeeze(pert_linf, dim=1)
               list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
               list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
               with torch.no_grad():
                  loss_val,_=loss(model,adv_images,labels)
               loss_avg=sum(loss_val)/len(loss_val)
               print('perturbation is: ', torch.t(pert_linf))
               print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
               #list_num.append(iter)
               list_loss+=loss_val.tolist()

    elif item['attack']=='MIFGSM':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.MIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'])  # torchattack
            adv_images,_, _, _, _, _ = atk(images, labels)
            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),
                                        dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1)
            list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
            list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
            with torch.no_grad():
                loss_val, labels_predict = loss(model, adv_images, labels)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            loss_avg = sum(loss_val) / len(loss_val)
            print('perturbation is: ', torch.t(pert_linf))
            print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
            # list_num.append(iter)
            list_loss += loss_val.tolist()
    elif item['attack'] == 'NIFGSM':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.NIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'],decay=item['decay'])  # torchattack
            adv_images,_, _, _, _, _ = atk(images, labels)
            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1)
            list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
            list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
            with torch.no_grad():
                loss_val, labels_predict = loss(model, adv_images, labels)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            loss_avg = sum(loss_val) / len(loss_val)
            print('perturbation is: ', torch.t(pert_linf))
            print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
            # list_num.append(iter)
            list_loss += loss_val.tolist()

    elif item['attack'] == 'BIM':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.BIM(model,eps=item['epsilon'],alpha=item['stepsize'],steps=item['iter_max'] )  # torchattack
            adv_images, perturbation, _, _, _, _ = atk(images, labels)
            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1)
            list_pert_l2 += (torch.squeeze(pert_l2, dim=1)).tolist()
            list_pert_l1 += torch.squeeze(pert_l1, dim=1).tolist()
            with torch.no_grad():
                loss_val, labels_predict = loss(model, adv_images, labels)
            success=(labels_predict!=labels)
            list_success_fail = list_success_fail + success.tolist()
            loss_avg = sum(loss_val) / len(loss_val)
            print('perturbation is: ', torch.t(pert_linf))
            print('avg_pert_linf is: ', pert_linf.sum() / len(pert_linf))
            # list_num.append(iter)
            list_loss += loss_val.tolist()
    elif item['attack'] == 'CW':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.CW(model, )  # torchattack
    end_time = time.time()

    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    list_success_sum=sum(list_success_fail)
    avg_loss=sum([val for val,con in zip(list_loss,list_success_fail) if con])/sum(list_success_fail)
    print('avg_loss is:',avg_loss)
    avg_pert_linf= sum([val for val, con in zip(list_pert_linf, list_success_fail) if con]) / list_success_sum
    avg_pert_l2 = sum([val for val, con in zip(list_pert_l2, list_success_fail) if con]) / list_success_sum
    avg_pert_l1 = sum([val for val, con in zip(list_pert_l1, list_success_fail) if con]) / list_success_sum
    print('avg_pert_linf is:',avg_pert_linf)
    print('avg_pert_l2 is:',avg_pert_l2 )
    print('avg_pert_l1 is:',avg_pert_l1)
    dict_save={'para':item,'time_used':time_used,'list_success':list_success_fail,'attack_success_rate':attack_success_rate,'list_loss':list_loss,'avg_loss':avg_loss,'avg_pert_linf':avg_pert_linf,'avg_pert_l2':avg_pert_l2,'avg_pert_l1':avg_pert_l1}
    if item['attack']=='Z_linf_M1':
        torch.save(dict_save,'./result/mnist/{}_{}_epsilon{}_stepsize{}_div{}_beta{}_rho{}_iter{}_es{}.pt'.format(item['model'],item['attack'],item['epsilon'],item['stepsize'],item['div'],item['beta'],item['rho'],item['iter_max'],item['early_stop']))
    elif item['attack']=='Z_linf_M2':
        torch.save(dict_save,'./result/mnist/{}_{}_epsilon{}_stepsize{}_div{}_beta{}_rho{}_iter{}_es{}.pt'.format(item['model'],item['attack'],item['epsilon'],item['stepsize'],item['div'],item['beta'],item['rho'],item['iter_max'],item['early_stop']))
    elif item['attack']=='PGD':
       torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_stepsize{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['stepsize'],item['iter_max']))
    elif item['attack']=='MIFGSM':
        torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_stepsize{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['stepsize'],item['iter_max']))
    elif item['attack']=='NIFGSM':
        torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_stepsize{}_iter{}_decay{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['stepsize'],item['iter_max'],item['decay']))
    elif item['attack']=='VNIFGSM':
        torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_iter{}_decay{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['stepsize'],item['iter_max'],item['decay']))
    elif item['attack']=='BIM':
        torch.save(dict_save,'./result/mnist/{}_attack_{}_epsilon{}_stepsize{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['stepsize'],item['iter_max']))

