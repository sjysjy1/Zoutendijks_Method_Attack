import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from zoutendijk_attack import zoutendijk_attack_Linfty_M1, zoutendijk_attack_Linfty_M2
import time
import matplotlib.pyplot as plt
import torchattacks
import random
from robustbench import load_model

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

test_dataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.ToTensor())

batch_size=32
list_para=[

{'model':'Standard','attack':'Z_linf_M1','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M1','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M1','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'div':10.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'div':10.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Standard','attack':'Z_linf_M2','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'div':10.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Standard','attack':'PGD','epsilon':8.0/255 ,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'PGD','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'PGD','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Standard','attack':'BIM','epsilon':8.0/255 ,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'BIM','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'BIM','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'MIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'NIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'NIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Standard','attack':'NIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},


{'model':'Wong2020Fast','attack':'Z_linf_M1','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Wong2020Fast','attack':'Z_linf_M1','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Wong2020Fast','attack':'Z_linf_M1','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.9,'early_stop':False},
{'model':'Wong2020Fast','attack':'Z_linf_M2','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Wong2020Fast','attack':'Z_linf_M2','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Wong2020Fast','attack':'Z_linf_M2','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.9,'batch_size':batch_size,'early_stop':False},
{'model':'Wong2020Fast','attack':'PGD','epsilon':8.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Wong2020Fast','attack':'PGD','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Wong2020Fast','attack':'PGD','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Wong2020Fast','attack':'BIM','epsilon':8.0/255 ,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'BIM','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'BIM','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'MIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'MIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'MIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'NIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'NIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Wong2020Fast','attack':'NIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},

{'model':'Salman2020','attack':'Z_linf_M1','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.99,'early_stop':False},
{'model':'Salman2020','attack':'Z_linf_M1','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.99,'early_stop':False},
{'model':'Salman2020','attack':'Z_linf_M1','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'batch_size':batch_size,'div':1.0,'beta':0.3,'rho': 0.99,'early_stop':False},
{'model':'Salman2020','attack':'Z_linf_M2','epsilon':8.0/255 ,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.99,'batch_size':batch_size,'early_stop':False},
{'model':'Salman2020','attack':'Z_linf_M2','epsilon':16.0/255,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.99,'batch_size':batch_size,'early_stop':False},
{'model':'Salman2020','attack':'Z_linf_M2','epsilon':32.0/255,'stepsize':0.007,'iter_max':20,'div':1.0,'beta':0.3,'rho': 0.99,'batch_size':batch_size,'early_stop':False},
{'model':'Salman2020','attack':'PGD','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Salman2020','attack':'PGD','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size,'norm':'Linf'},
{'model':'Salman2020','attack':'BIM','epsilon':8.0/255 ,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'BIM','epsilon':16.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'BIM','epsilon':32.0/255,'stepsize':0.01,'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'MIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'MIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'MIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'NIFGSM','epsilon':8.0/255 ,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'NIFGSM','epsilon':16.0/255,'stepsize':0.007,'decay':0.3, 'iter_max':20,'batch_size':batch_size},
{'model':'Salman2020','attack':'NIFGSM','epsilon':32.0/255,'stepsize':0.007,'decay':0.3,'iter_max':20,'batch_size':batch_size},
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
    images_labels = torch.load('imagenet_first_1000.pt')
    test_data_normalized = images_labels['imgs']
    test_labels = torch.tensor(images_labels['labels']).to(device)

    if item['model'] == 'Standard':
        model = load_model(model_name='Standard_R50',dataset='imagenet', norm='Linf')
    elif item['model'] == 'Wong2020Fast':
        model = load_model(model_name='Wong2020Fast',dataset='imagenet', norm='Linf')
    elif item['model'] == 'LiuSwinL':
        model = load_model(model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', norm='Linf')
    elif item['model'] == 'LiuConvNeXtL':
        model = load_model(model_name='Liu2023Comprehensive_ConvNeXt-L', dataset='imagenet', norm='Linf')
    elif item['model'] == 'Salman2020':
        model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', norm='Linf')
    elif item['model'] == 'Singh2023':
        model = load_model(model_name='Singh2023Revisiting_ConvNeXt-L-ConvStem', dataset='imagenet', norm='Linf')



    model.to(device)
    model.eval()  # turn off the dropout
    test_accuracy = False
    if test_accuracy == True:
        predict_result = torch.tensor([], device=device)
        for i in range(50):
            outputs = model(test_data_normalized[20 * i:20 * i + 20].to(device))
            _, labels_predict = torch.max(outputs, 1)
            predict_result = torch.cat((predict_result, labels_predict), dim=0)
        correct = torch.eq(predict_result, test_labels)
        #imshow(torchvision.utils.make_grid(images_test[0].cpu().data, normalize=True),'Predict:{}'.format(predict_result[0]))
        torch.save(correct, './result/imagenet-first1000/{}_Imagenet_correct_predict.pt'.format(item['model']))
    else:
        correct = torch.load('./result/imagenet-first1000/{}_Imagenet_correct_predict.pt'.format(item['model']))


    correct_sum = correct.sum()
    clean_accuracy = correct_sum / 1000.0
    #correct_sum = correct[0:1000].sum()
    print('model clean accuracy:', clean_accuracy)
    correct_index=[]
    for i in range(1000):
        if correct[i]:
            correct_index.append(i)

    start_time = time.time()
    if item['attack']=='Z_linf_M1':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i:i + item['batch_size']]]

            #for showing image
#            outputs = model(images)
#            _, labels_predict = torch.max(outputs, 1)
#            for i in range(9):
#                # define subplot
#                plt.subplot(330 + 1 + i)
#                # plot raw pixel data
#                plt.imshow(images[i], cmap=plt.get_cmap('gray'))
#            # show the figure
#            plt.show()
            adv_images=zoutendijk_attack_Linfty_M1(model,images,labels,epsilon=item['epsilon'],iter_max=item['iter_max'],stepsize_default=item['stepsize'],beta=item['beta'],rho=item['rho'],div=item['div'])
            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1).tolist()  #if .tolist() is not used, then GPU memory will keep increasing
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
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i:i + item['batch_size']]]

            # for showing image
#            outputs = model(images)
#            _, labels_predict = torch.max(outputs, 1)
            #            for i in range(9):
            #                # define subplot
            #                plt.subplot(330 + 1 + i)
            #                # plot raw pixel data
            #                plt.imshow(images[i], cmap=plt.get_cmap('gray'))
            #            # show the figure
            #            plt.show()
            adv_images = zoutendijk_attack_Linfty_M2(model, images, labels, epsilon=item['epsilon'],iter_max=item['iter_max'], stepsize_default=item['stepsize'],beta=item['beta'], rho=item['rho'], div=item['div'])

            #        outputs = model(adv_images)
    #        _, labels_predict = torch.max(outputs, 1)
    #        for i in range(9):
    #            # define subplot
    #            plt.subplot(330 + 1 + i)
    #            # plot raw pixel data
    #            plt.axis('off')
    #            plt.imshow(adv_images[i].cpu().data.reshape(28,28), cmap=plt.get_cmap('gray'))
    #            plt.title('Predict:{}'.format(labels_predict[i].item()),fontsize='x-small',x=0.5,y=0.94)
#
    #            plt.subplots_adjust(wspace=0.01, hspace=0.1)
    #        # show the figure
    #        plt.show()
    #        plt.subplots_adjust(wspace=0.1, hspace=0.01)

            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1).tolist()
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
               images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
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
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.MIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'],decay=item['decay'])  # torchattack
            adv_images,_,_,_,_,_= atk(images, labels)

            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf, dim=1).tolist()  #if .tolist() is not used, then GPU memory will keep increasing
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

    elif item['attack'] == 'NIFGSM':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.NIFGSM(model, eps=item['epsilon'], alpha=item['stepsize'],steps=item['iter_max'],decay=item['decay'])  # torchattack
            adv_images, _, _, _, _, _ = atk(images, labels)

            pert_linf = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1),
                                        dim=1)
            pert_l2 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1), dim=1)
            pert_l1 = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1), dim=1)

            list_pert_linf += torch.squeeze(pert_linf,
                                            dim=1).tolist()  # if .tolist() is not used, then GPU memory will keep increasing
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
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].to(device)
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
    elif item['attack'] == 'FAB':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.FAB(model, )  # torchattack
            adv_image, perturbation, num, success, loss, predict_label = atk(image, label)
            perturbation = torch.unsqueeze(torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1), dim=1)
            outputs = model(images)
            _, labels_predict = torch.max(outputs, 1)
            success = ~(labels_predict == labels)
            list_success_fail = list_success_fail + success.tolist()
            list_pert = list_pert + torch.squeeze(perturbation, dim=1).tolist()
            print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
            list_num.append(iter)
            list_loss.append(loss.item())
    elif item['attack'] == 'CW':
        for i in range(0, len(correct_index), item['batch_size']):
            print('***************{}th batch***********'.format(int(i / item['batch_size'])))
            images = test_data_normalized[correct_index[i:i + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i:i + item['batch_size']]]
            atk = torchattacks.CW(model, )  # torchattack
    end_time = time.time()

    attack_success_rate=sum(list_success_fail)/len(correct_index)
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
    print('avg_pert_l2 is:',avg_pert_l2)
    print('avg_pert_l1 is:',avg_pert_l1)
    dict_save={'para':item,'time_used':time_used,'list_success':list_success_fail,'attack_success_rate':attack_success_rate,'list_loss':list_loss,'avg_loss':avg_loss,'avg_pert_linf':avg_pert_linf,'avg_pert_l2':avg_pert_l2,'avg_pert_l1':avg_pert_l1}
    if item['attack']=='Z_linf_M1':
        torch.save(dict_save,'./result/imagenet-first1000/{}_{}_epsilon{}_stepsize{}_div{}_beta{}_rho{}_iter{}_es{}.pt'.format(item['model'],item['attack'],item['epsilon'],item['stepsize'],item['div'],item['beta'],item['rho'],item['iter_max'],item['early_stop']))
    elif item['attack']=='Z_linf_M2':
        torch.save(dict_save,'./result/imagenet-first1000/{}_{}_epsilon{}_stepsize{}_div{}_beta{}_rho{}_iter{}_es{}.pt'.format(item['model'],item['attack'],item['epsilon'],item['stepsize'],item['div'],item['beta'],item['rho'],item['iter_max'],item['early_stop']))
    elif item['attack']=='PGD':
       torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_epsilon{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['iter_max']))
    elif item['attack']=='MIFGSM':
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_epsilon{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['iter_max']))
    elif item['attack']=='NIFGSM':
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_epsilon{}_iter{}_decay{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['iter_max'],item['decay']))
    elif item['attack']=='BIM':
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_epsilon{}_iter{}.pt'.format(item['model'], item['attack'], item['epsilon'],item['iter_max']))

