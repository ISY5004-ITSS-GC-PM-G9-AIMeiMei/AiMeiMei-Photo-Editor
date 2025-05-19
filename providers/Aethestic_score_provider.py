import os
import torch
import torchvision
import numpy as np
import cv2
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .AIMeiMei_FID.model.backbone import resnet50_backbone
from .AIMeiMei_FID.model.model_main import IQARegression
from .AIMeiMei_FID.option.config import Config
# from AIMeiMei_FID.model.backbone import resnet50_backbone
# from AIMeiMei_FID.model.model_main import IQARegression
# from AIMeiMei_FID.option.config import Config


class AestheticScorer:
    def __init__(self, checkpoint_path, gpu_id=0):
        
        self.config = Config({
            'gpu_id': gpu_id,
            'n_enc_seq': 32 * 24 + 12 * 9 + 7 * 5,
            'n_layer': 14,
            'd_hidn': 384,
            'batch_size': 1,
            'i_pad': 0,
            'd_ff': 384,
            'd_MLP_head': 1152,
            'n_head': 6,
            'd_head': 384,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'layer_norm_epsilon': 1e-12,
            'n_output': 1,
            'Grid': 10,
            'scale_1': 384,
            'scale_2': 224,
        })
        self.config.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model_backbone = resnet50_backbone().to(self.config.device)
        self.model_transformer = IQARegression(self.config).to(self.config.device)

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
        self.model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
        self.model_backbone.eval()
        self.model_transformer.eval()

        # 预处理 pipeline
        self.transforms = torchvision.transforms.Compose([
            self.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            self.ToTensor()
        ])

    class Normalize:
        def __init__(self, mean, var):
            self.mean = mean
            self.var = var

        def __call__(self, sample):
            sample[:, :, 0] = (sample[:, :, 0] - self.mean[0]) / self.var[0]
            sample[:, :, 1] = (sample[:, :, 1] - self.mean[1]) / self.var[1]
            sample[:, :, 2] = (sample[:, :, 2] - self.mean[2]) / self.var[2]
            return sample

    class ToTensor:
        def __call__(self, sample):
            sample = np.transpose(sample, (2, 0, 1))
            return torch.from_numpy(sample)

    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w, _ = img.shape

        # 多尺度输入
        scale_1 = cv2.resize(img, dsize=(self.config.scale_1, int(h * (self.config.scale_1 / w))))
        scale_2 = cv2.resize(img, dsize=(self.config.scale_2, int(h * (self.config.scale_2 / w))))
        scale_2 = scale_2[:160, :, :]

        # 转换
        img_tensor = self.transforms(img).to(self.config.device).unsqueeze(0)
        scale_1_tensor = self.transforms(scale_1).to(self.config.device).unsqueeze(0)
        scale_2_tensor = self.transforms(scale_2).to(self.config.device).unsqueeze(0)

        mask_inputs = torch.ones(1, self.config.n_enc_seq + 1).to(self.config.device)

        # 推理
        with torch.no_grad():
            feat1 = self.model_backbone(img_tensor)
            feat2 = self.model_backbone(scale_1_tensor)
            feat3 = self.model_backbone(scale_2_tensor)
            pred = self.model_transformer(mask_inputs, feat1, feat2, feat3)

        return float(pred.item())
    
# # 初始化（只需一次）
# scorer = AestheticScorer(
#     checkpoint_path="/media/labpc2x2080ti/data/Mohan_Workspace/AiMeiMei-Photo-Editor/providers/AIMeiMei_FID/model/epoch100.pth",
#     gpu_id=0  # 可改为 -1 用 CPU
# )

# # 推理单张图片"
# score = scorer.predict("/media/labpc2x2080ti/data/Mohan_Workspace/AiMeiMei-Photo-Editor/providers/AIMeiMei_FID/test_image/3039024.jpg")
# print(f"Aesthetic score: {score:.4f}")
