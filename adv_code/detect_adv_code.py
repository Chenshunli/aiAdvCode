import os
import torch
import requests
import time
import torch.nn as nn
from torchvision.models import mobilenet_v2,resnet18
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd
from robust_layer import GradientConcealment, ResizedPaddingLayer
from timm.models import create_model

from advertorch.attacks import LinfPGDAttack
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc

device = "cuda" if torch.cuda.is_available() else "cpu"


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())


normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

imagenet_label2classname = ImageNetClassNameLookup()


### 常规模型加载
class Model(nn.Module):
    def __init__(self, l=290):
        super(Model, self).__init__()

        self.l = l
        self.gcm = GradientConcealment()
        #model = resnet18(pretrained=True)
        model = mobilenet_v2(pretrained=True)

        # pth_path = "/Users/rocky/Desktop/训练营/model/mobilenet_v2-b0353104.pth"
        # print(f'Loading pth from {pth_path}')
        # state_dict = torch.load(pth_path, map_location='cpu')
        # is_strict = False
        # if 'model' in state_dict.keys():
        #    model.load_state_dict(state_dict['model'], strict=is_strict)
        # else:
        #    model.load_state_dict(state_dict, strict=is_strict)

        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = nn.Sequential(normalize, model)

    def load_params(self):
        pass

    def forward(self, x):
        #x = self.gcm(x)
        #x = ResizedPaddingLayer(self.l)(x)
        out = self.model(x)
        return out


### 对抗攻击监测模型
class Detect_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Detect_Model, self).__init__()
        self.num_classes = num_classes
        #model = create_model('mobilenetv3_large_075', pretrained=False, num_classes=num_classes)
        model = create_model('resnet50', pretrained=False, num_classes=num_classes)

        # self.multi_PreProcess = multi_PreProcess()
        pth_path = os.path.join("/home/Lesson5_code/model", 'track2_resnet50_ANT_best_albation1_64_checkpoint.pth')
        #pth_path = os.path.join("/Users/rocky/Desktop/训练营/Lesson5_code/model/", "track2_tf_mobilenetv3_large_075_64_checkpoint.pth")
        state_dict = torch.load(pth_path, map_location='cpu')
        is_strict = False
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict['model'], strict=is_strict)
        else:
            model.load_state_dict(state_dict, strict=is_strict)
        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.model = nn.Sequential(normalize, self.multi_PreProcess, model)
        self.model = nn.Sequential(normalize, model)

    def load_params(self):
        pass

    def forward(self, x):
        # x = x[:,:,32:193,32:193]
        # x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=True)
        # x = self.multi_PreProcess.forward(x)
        out = self.model(x)
        if self.num_classes == 2:
            out = out.softmax(1)
            #return out[:,1:]
            return out[:,1:]


model = Model().eval().to(device)

detect_model = Detect_Model().eval().to(device)


### 读取图片
def get_image():
        img_path = os.path.join("/home/Lesson5_code/adv_code/orig_images", "vid_5_31040.jpg_3.jpg")
        # img_path = os.path.join("/home/Lesson5_code/adv_code/adv_results", "adv_image.png")
        img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

        if os.path.exists(img_path):
            return _load_image(img_path)
        else:
            import urllib
            urllib.request.urlretrieve(img_url, img_path)
            return _load_image(img_path)

def _load_image(img_path):
    from skimage.io import imread
    return imread(img_path) / 255.


model = Model().eval().to(device)
detect_model = Detect_Model().eval().to(device)


if __name__ == "__main__":

    ## 读取图片
    np_img = get_image()
    img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)

    ### 对抗攻击监测
    detect_pred = detect_model(img)
    x = detect_pred.tolist()[0][0]
    ### 对抗攻击监测结果判断，如果风险，则报警，否则进一步进行后续业务（常规模型对样本进行分类）
    if detect_pred > 0.5:
        id = 't50SOmT'
        # 填写喵提醒中，发送的消息，这里放上前面提到的图片外链
        text = "出现对抗攻击风险！！"
        print(text)
        print(image_name)
        # print("结果概率：")
        # print("%.2f" % x)
        print("\n")
        ts = str(time.time())  # 时间戳
        type = 'json'  # 返回内容格式
        request_url = "http://miaotixing.com/trigger?"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

        result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=" + type,
                            headers=headers)
    else:
        print("正常样本")
        print(image_name)
        # print("结果概率：")
        # print("%.2f" % x)
        ### 正常样本分类
        pred = imagenet_label2classname(predict_from_logits(model(img)))
        print("预测结果：")
        print(pred)
