import model_utils.kernel_processer as k_processor
import torch
import torch.nn.functional as F
import numpy as np
import math
from model_utils import ndcg

class regression_processer(k_processor.kernel_processer):
    def __init__(self,grad_clip=False,clip_value=5.0,epoch_lr_decay=250,
    classify_mode=False,loss_constrain=False,constrain_value=1.0):
        super(regression_processer, self).__init__()
        self.loss_function=torch.nn.MSELoss()
        self.bce_loss=torch.nn.BCELoss()
        self.grad_clip=grad_clip
        self.clip_value=clip_value
        self.epoch_lr_decay=epoch_lr_decay
        self.classify_mode=classify_mode
        self.loss_constrain=loss_constrain
        self.constrain_value=constrain_value
        self.pred1=None
        self.pred=None
        self.label=None

    def train(self,step,data):
        optimizer=self.optimizers[0]
        model=self.models[0]
        total_loss={}
        x1=data[0]
        x2=data[1]
        y=data[3]
        score1,score2,out2,M2,score3,out3,M3= model(x1, x2)
        score = np.array(score2, score3)
        out = np.array(out2, out3)
        loss_pred = self.loss_function(score, y)
        loss = None
        if (len(out2.size()) != len(M2.size())) or (len(out3.size()) != len(M3.size())):
            loss = self.loss_function(out, y) + loss_pred
        else:
            if (self.classify_mode):
                label_c = (y > 0.5).float()
                out_logit = (score > 0.5).float()
                mask = 1 - (label_c == out_logit).float()
                loss = loss_pred + self.constrain_value * torch.mean(mask * (score - y) * (score - y))
            else:
                loss = loss_pred
        loss.backward()
        optimizer.step()
        self.zero_grad_for_all()
        mse = (score - y) * (score - y)
        mse = torch.mean(mse, 0)
        mse = mse.detach().cpu().numpy()
        if(step%10==0):
            total_loss["train_loss"]=loss.detach().cpu().item()
            total_loss["train_loss_pred"]=loss_pred.detach().cpu().item()
        return total_loss

    def evaluate(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        evaluate_dict={}
        x1=data[0]
        x2=data[1]
        y=data[3]
        score1,score2,out2,M2,score3,out3,M3= model(x1, x2)
        # out=F.sigmoid(out)
        score = np.array(score2, score3)
        out = np.array(out2, out3)
        loss_pred = self.loss_function(score, y)
        loss = None
        if (len(out2.size()) != len(M2.size())) or (len(out3.size()) != len(M3.size())):
            loss = self.loss_function(out, y) + loss_pred
            else:
            if (self.classify_mode):
                label_c = (y > 0.5).float()
                out_logit = (score > 0.5).float()
                mask = 1 - (label_c == out_logit).float()
                loss = loss_pred + self.constrain_value * torch.mean(mask * (score - y) * (score - y))
            else:
                loss = loss_pred
        mse = (score - y) * (score - y)
        mse = torch.mean(mse, 0)
        mse = mse.detach().cpu().numpy()

        evaluate_dict["test_loss"]=loss.detach().cpu().item()
        evaluate_dict["test_loss_pred"]=loss_pred.detach().cpu().item()
        return x1.size(0),evaluate_dict["test_loss_pred"],evaluate_dict

    def test(self,step,data):
        data=self.tencrop_process(data)
        model=self.models[0]
        evaluate_dict={}
        x1=data[0]
        x2=data[1]
        y=data[3]
        score1,score2,out2,M2,score3,out3,M3= model(x1, x2)
        # out=F.sigmoid(out)
        score = np.array(score2,score3)
        out = np.array(out2,out3)
        loss_pred=self.loss_function(score,y)
        loss=None
        if(len(out2.size())!=len(M2.size())) or (len(out3.size())!=len(M3.size())):
            loss=self.loss_function(out,y)+loss_pred
        else:
            if(self.classify_mode):
                label_c=(y>0.5).float()
                out_logit=(score>0.5).float()
                mask=1-(label_c==out_logit).float()
                loss=loss_pred+self.constrain_value*torch.mean(mask*(score-y)*(score-y))
            else:
                loss=loss_pred
        mse=(score-y)*(score-y)
        mse=torch.mean(mse,0)
        mse=mse.detach().cpu().numpy()
        if(self.pred is None):
            self.pred1=np.array(score1,score2,score3).detach().cpu().numpy()
            self.pred=score.detach().cpu().numpy()
            self.label=y.detach().cpu().numpy()
        else:
            self.pred=np.concatenate((self.pred,out.detach().cpu().numpy()),axis=0)
            self.label=np.concatenate((self.label,y.detach().cpu().numpy()),axis=0)

        evaluate_dict["test_loss"]=loss.detach().cpu().item()
        evaluate_dict["test_loss_pred"]=loss_pred.detach().cpu().item()
        return x1.size(0),evaluate_dict["test_loss_pred"],evaluate_dict

    def on_finish(self):
        d={}
        mse=(self.pred-self.label)*(self.pred-self.label)
        mse=np.mean(mse,axis=0)
        d["mse_1"]=mse[0]
        d["mse_2"]=mse[1]
        avg_label=np.mean(self.label,axis=0).reshape((1,2))
        mse_y=(self.label-avg_label)*(self.label-avg_label)
        mse_y=np.mean(mse_y,axis=0)
        r_sq=1-mse/mse_y
        d["r_square_1"]=r_sq[0]
        d["r_square_2"]=r_sq[1]
        ndcg_sore = ndcg.ndcg(self.label, self.pred)
        d["NDCG"] = ndcg_sore[1]
        return d


    def update_optimizers(self,epoch,step,total_data_numbers):
        optimizer=self.optimizers[0]
        if(epoch==self.epoch_lr_decay and step==0):
            print("change the learning rate ")
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.1




def optimizers_producer(models,lr_base,lr_fc,weight_decay,paral=True):
    optimizers=[]
    model=models[0]
    c_param=None
    if(paral):
        c_param=model.module.fc.parameters()
    else:
        c_param=model.fc.parameters()

    clssify_params = list(map(id, c_param))
    base_params = filter(lambda p: id(p) not in clssify_params,model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params,'lr':lr_base},
        {'params': c_param, 'lr': lr_fc},
        ], lr_base, momentum=0.9, weight_decay=weight_decay)
    optimizers.append(optimizer)
    return optimizers


def optimizers_producer_classify(models,lr_base,lr_fc,weight_decay,paral=True):
    optimizers=[]
    model=models[0]
    c_param=None
    classify_param=None
    if(paral):
        c_param=model.module.fc.parameters()
        classify_param=model.module.fc_classify.parameters()
    else:
        c_param=model.fc.parameters()
        classify_param=model.fc_classify.parameters()

    c_params = list(map(id, c_param))
    classify_params=list(map(id,classify_param))
    base_params = filter(lambda p: id(p) not in classify_params+c_params,model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params,'lr':lr_base},
        {'params': c_param, 'lr': lr_fc},
        {'params': classify_param, 'lr': lr_fc},
        ], lr_base, momentum=0.9, weight_decay=weight_decay)
    optimizers.append(optimizer)
    return optimizers
