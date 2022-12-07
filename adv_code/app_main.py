# aidlux相关
from cvs import *
import time
import cv2
import torch
import requests
import aidlite_gpu
import torch.nn as nn
import torchvision.utils
import copy

from detect_adv_code import Model, Detect_Model
from torchvision.models import mobilenet_v2, resnet18
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import ImageNetClassNameLookup
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch_examples.utils import bhwc2bchw
from advertorch_examples.utils import bchw2bhwc
from advertorch.attacks import FGSM, LinfPGDAttack
from extractUtil import detect_postprocess, preprocess_img, draw_detect_res, extract_detect_res

device = "cuda" if torch.cuda.is_available() else "cpu"

normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imagenet_label2classname = ImageNetClassNameLookup()

### 对抗攻击常规模型加载
model = mobilenet_v2(pretrained=True)
model.eval()
model = nn.Sequential(normalize, model)
model = model.to(device)

### 对抗攻击替身模型加载
model_su = resnet18(pretrained=True)
model_su.eval()
model_su = nn.Sequential(normalize, model_su)
model_su = model_su.to(device)

def PGDAttack(model, model_su, img_np):
    """
    args:
       model-常规模型
       model_su-替身模型
       img_np - 原始图片
    returns:
        advimg - 增加对抗攻击后的图片
    """
    np_img = img_np[:,:,::-1] / 255.0
    img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
    imagenet_label2classname = ImageNetClassNameLookup()
    ### 测试模型输出结果
    pred = imagenet_label2classname(predict_from_logits(model(img)))
    print("test output:", pred)
    ### 输出原label
    pred_label = predict_from_logits(model_su(img))
    ### 对抗攻击：PGD攻击算法
    # adversary = LinfPGDAttack(
    #     model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
    #     rand_init=True, targeted=False)
    ## 对抗攻击：FGSM算法
    adversary = FGSM(
        model_su, eps=20/255, targeted=False)
    advimg = adversary.perturb(img, pred_label)
    # advimg = np.transpose(advimg.squeeze().numpy(), (1, 2, 0))
    return advimg

flag_attack = 0
model_det = Model().eval().to(device)
model_det_attack = Detect_Model().eval().to(device)

# AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
aidlite = aidlite_gpu.aidlite()
# Aidlite模型路径
model_path = '/home/Lesson5_code/yolov5_code/models/yolov5_car_best-fp16.tflite'
# 定义输入输出shape
in_shape = [1 * 640 * 640 * 3 * 4]
out_shape = [1 * 25200 * 6 * 4]
# 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)
# 读取图片进行推理
# 设置测试集路径
source = "/home/Lesson5_code/adv_code/test_images"
images_list = os.listdir(source)
print(images_list)

frame_id = 0
# 读取数据集
for image_name in images_list[:1]:
    frame_id += 1
    print("frame_id:", frame_id)
    image_path = os.path.join(source, image_name)
    frame = cvs.imread(image_path)
    # 预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
    aidlite.setInput_Float32(img, 640, 640)
    # 模型推理API
    aidlite.invoke()
    # 读取返回的结果
    pred = aidlite.getOutput_Float32(0)
    # 数据维度转换
    pred = pred.reshape(1, 25200, 6)[0]
    # 模型推理后处理
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.25, iou_thres=0.45)
    all_boxes = pred[0]
    frame = frame.astype(np.uint8)
    if len(all_boxes) > 0:
        for box in all_boxes:
            x, y, w, h = [int(t) for t in box[:4]]
            cut_img = frame[y:(y+h), x:(x + w)]
            # print("---", cut_img.shape)
            if flag_attack == 1:
                advimg = PGDAttack(model, model_su, cut_img)
                print("+++++", type(advimg), advimg.shape)
            else:
                cut_img = copy.deepcopy(cut_img[:,:,::-1] / 255)
                advimg = torch.tensor(bhwc2bchw(cut_img))[None, :, :, :].float().to(device)
            ### 无对抗攻击监测模型
            # detect_pred = model_det(advimg)
            ### 对抗攻击监测
            detect_pred = model_det_attack(advimg)
            # img = torch.tensor(bhwc2bchw(np_img))[None, :, :, :].float().to(device)
            print("****", detect_pred)
            if detect_pred > 0.5:
                id = 't50SOmT'
                # 填写喵提醒中，发送的消息，这里放上前面提到的图片外链
                text = "出现对抗攻击风险！！"
                print("++++", text)
                ts = str(time.time())  # 时间戳
                type = 'json'  # 返回内容格式
                request_url = "http://miaotixing.com/trigger?"

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

                result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=" + type,
                                    headers=headers)
            else:
                pred = imagenet_label2classname(predict_from_logits(model(advimg)))
                print(pred)

