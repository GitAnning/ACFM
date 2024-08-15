#change the newwork structure
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class resnet_base(nn.Module):
    def __init__(self):
        super(resnet_base,self).__init__()
        self.base=models.resnet101(pretrained=True)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out=x
        return torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)

class Baseline(nn.Module):
    def __init__(self,C):
        super(Baseline,self).__init__()
        self.base=models.resnet101(pretrained=True)
        self.fc=nn.Linear(2048,3)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out=x
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        out=self.fc(out)
        return out,out,out

class SENet_block(nn.Module):
    def __init__(self,activate_type="none"):
        super(SENet_block,self).__init__()
        self.conv1=nn.Conv2d(2048,2048,1)
        self.conv2=nn.Conv2d(2048,2048,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.conv_classify=nn.Conv2d(2048,2048,1)
        self.activate_type=activate_type

    def forward(self,x):
        out1=self.conv1(x)
        out1=self.sigmoid(out1)
        out1=torch.mean(out1.view(-1,out1.size(1),out1.size(2)*out1.size(3)),2)
        out_channel_wise=None
        if(self.activate_type=="none"):
            out_channel_wise=out1
        if(self.activate_type=="softmax"):
            out_channel_wise=F.softmax(out1,dim=1)
        if(self.activate_type=="sigmoid"):
            out_channel_wise=F.sigmoid(out1)
        if(self.activate_type=="sigmoid_res"):
            out_channel_wise=F.sigmoid(out1)+1
        out2=self.conv2(x)
        out2=self.relu(out2)
        out=out2*out_channel_wise.view(-1,out1.size(1),1,1)
        return out,out_channel_wise,out2

class spatial_block(nn.Module):
    def __init__(self):
        super(spatial_block,self).__init__()
        self.tanh=nn.Tanh()
        self.fc=nn.Linear(2048,2048)
        self.conv=nn.Conv2d(2048,2048,1)
        self.conv2=nn.Conv2d(2048,1,1)

    def forward(self,x,channel_wise):
        out=self.conv(x)
        if(len(channel_wise.size())!=len(x.size())):
            channel_wise=self.fc(channel_wise)
            channel_wise=channel_wise.view(-1,channel_wise.size(1),1,1)
            out=out+channel_wise
        out=self.tanh(out)
        out=self.conv2(out)
        x_shape=out.size(2)
        y_shape=out.size(3)
        out=out.view(-1,x_shape*y_shape)
        out=F.softmax(out,dim=1)
        out=out.view(-1,1,x_shape,y_shape)
        out=x*out
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        return out

class att_block(nn.Module):
    def __init__(self):
        super(att_block,self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=2048, num_heads=3)

    def forward(self, img_feature, text_feature):
        batch_size, num_channels, height, width = self.image_feature.size()
        img_features = img_feature.view(batch_size, num_channels, height * width)
        img_features = img_feature.permute(0, 2, 1)
        text_features = text_feature.unsqueeze(1)
        query = img_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1, 0, 2).squeeze(1)
        return attn_output, attn_output_weights

class SENet_channel_wise_with_attention(nn.Module):
    def __init__(self,C,activate_type="none"):
        super(SENet_channel_wise_with_attention,self).__init__()
        self.se_block=None
        self.se_block=SENet_block(activate_type)
        self.fc=nn.Linear(2048*2,3)
        self.fc_classify=nn.Linear(2048*2,C)
        self.spatial=spatial_block()

    def forward(self,x):
        out,out_channel_wise,out2=self.se_block.forward(x)
        out=torch.mean(out.view(-1,out.size(1),out.size(2)*out.size(3)),2)
        spatial_feature=self.spatial.forward(out2,out_channel_wise)
        feature_cat=torch.cat((out,spatial_feature),1)
        out=self.fc(feature_cat)
        out_classify=self.fc_classify(feature_cat)
        return out,out_classify,out

class Imagery_block(nn.Module):
    def __init__(self,C,activate_type="none"):
        super(Imagery_block,self).__init__()
        self.att_block =att_block()
        self.hidden1 = nn.Linear(2048, 3000)
        self.hidden2 = nn.Linear(3000, 1000)
        self.hidden3 = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 1)
        self.fc_classify = nn.Linear(1000, C)
    def forward(self, x1, x2):
        attn_output, attn_output_weights = self.att_block.forward(x1, x2)
        score = nn.ReLU(self.hidden1(attn_output))
        score = nn.ReLU(self.hidden2(score))
        score = nn.ReLU(self.hidden3(score))
        out_classify3 = self.fc_classify(score)
        score = F.softmax(self.hidden4(score), dim=1)
        return score, out_classify, score

class ACFM(nn.Module):
    def __init__(self, C, activate_type="none"):
        self.base = models.resnet101(pretrained=True)
        self.VAD_block = SENet_channel_wise_with_attention(C, activate_type)
        self.Imagery_block = Imagery_block(C, activate_type)
        self.image_feature = None
        self.text_feature = None
        self.att_block =att_block()
        self.hidden1 = nn.Linear(2048+2, 3000)
        self.hidden2 = nn.Linear(3000, 1000)
        self.hidden3 = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 1)
        self.fc_classify = nn.Linear(1000, C)

    def forward(self,x1,x2):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            self.image_feature = module(x1)
            self.text_feature = nn.Linear(x2, 2048)
        VAD_score,out_classify1,out1 = self.VAD_block.forward(self.image_feature)
        Imagery_score,out_classify2,out2 = self.Imagery_block.forward(self.image_feature,self.text_feature_feature)
        attn_output, attn_output_weights = self.att_block.forward(self.image_feature,self.text_feature)
        input_feature = torch.cat((attn_output,VAD_score,Imagery_score),1)
        energy = nn.ReLU(self.hidden1(input_feature))
        energy = nn.ReLU(self.hidden2(energy))
        energy = nn.ReLU(self.hidden3(energy))
        out_classify3 = self.fc_classify(energy)
        energy = F.softmax(self.hidden4(energy),dim=1)

        return VAD_score,Imagery_score,out_classify2,out2,energy,out_classify3,energy


