import os
import torch
import torch.nn as nn
import torchvision.utils
from torchvision.models import mobilenet_v2,resnet18
from advertorch.utils import predict_from_logits
from advertorch.utils import NormalizeByChannelMeanStd
from robust_layer import GradientConcealment, ResizedPaddingLayer
from timm.models import create_model

from advertorch.attacks import LinfPGDAttack,L1PGDAttack,L2PGDAttack, LinfBasicIterativeAttack
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc

device = "cuda" if torch.cuda.is_available() else "cpu"


### 读取图片
def get_image():
    img_path = os.path.join("/home/Lesson5_code/adv_code/orig_images", "vid_5_31040.jpg_3.jpg")
    img_url = "https://farm1.static.flickr.com/230/524562325_fb0a11d1e1.jpg"

    def _load_image():
        from skimage.io import imread
        return imread(img_path) / 255.

    if os.path.exists(img_path):
        return _load_image()
    else:
        import urllib
        urllib.request.urlretrieve(img_url, img_path)
        return _load_image()


def tensor2npimg(tensor):
    return bchw2bhwc(tensor[0].cpu().numpy())

### 展示攻击结果
def show_images(model, img, advimg, enhance=127):
    np_advimg = tensor2npimg(advimg)
    np_perturb = tensor2npimg(advimg - img)

    pred = imagenet_label2classname(predict_from_logits(model(img)))
    advpred = imagenet_label2classname(predict_from_logits(model(advimg)))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np_img)

    plt.axis("off")
    plt.title("original image\n prediction: {}".format(pred))
    plt.subplot(1, 3, 2)
    # plt.imshow(np_perturb * enhance + 0.5)

    plt.axis("off")
    plt.title("the perturbation,\n enhanced {} times".format(enhance))
    plt.subplot(1, 3, 3)
    # plt.imshow(np_advimg)
    # plt.imshow(np_advimg.astype(np.uint8))
    plt.axis("off")
    plt.title("perturbed image\n prediction: {}".format(advpred))
    plt.show()


normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


### 常规模型加载
model = mobilenet_v2(pretrained=True)
model.eval()
model = nn.Sequential(normalize, model)
model = model.to(device)


### 替身模型加载
model_su = resnet18(pretrained=True)
model_su.eval()
model_su = nn.Sequential(normalize, model_su)
model_su = model_su.to(device)


### 数据预处理
np_img = get_image()
img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
imagenet_label2classname = ImageNetClassNameLookup()


### 测试模型输出结果
pred = imagenet_label2classname(predict_from_logits(model(img)))
print("test output:", pred)


### 输出原label
pred_label = predict_from_logits(model_su(img))


### 对抗攻击：PGD攻击算法
adversary = LinfPGDAttack(
   model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)

## 对抗攻击：L1PGD攻击算法 (eps = 100, 400, 1600)
# adversary2 = LinfBasicIterativeAttack(
#    model_su, eps=160, eps_iter=2/255, nb_iter=80,
#    rand_init=True, targeted=False)


### 对抗攻击：L2PGD攻击算法 (eps = 0.5, 2, 8)
adversary3 = L2PGDAttack(
   model_su, eps=0.5, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)


### 对抗攻击：LinfMomentumIterativeAttack攻击算法 (eps = 0.5/255, 2/255, 8/255)
# adversary4 = LinfMomentumIterativeAttack(
#    model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
#    rand_init=True, targeted=False)



### 完成攻击，输出对抗样本
advimg = adversary.perturb(img, pred_label)


### 展示源图片，对抗扰动，对抗样本以及模型的输出结果
show_images(model, img, advimg)


### 迁移攻击样本保存
# save_path = "/home/Lesson5_code/adv_code/adv_results/"
# torchvision.utils.save_image(advimg.cpu().data, save_path + "adv_image.png")

