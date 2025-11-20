import torch
import torch.nn as nn


class ImgNet_T(nn.Module):
    """图像编码网络backbone"""
    def __init__(self, code_len=64):
        super(ImgNet_T, self).__init__()

        self.img_encoder1 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.img_encoder2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.img_encoder3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.imgHashing = nn.Sequential(
            nn.Linear(512, code_len),
        )

    def forward(self, x):
        feat1 = self.img_encoder1(x)
        feat2 = self.img_encoder2(feat1)
        feat3 = self.img_encoder3(feat2)
        code = self.imgHashing(feat3)
        code = torch.tanh(code)
        return feat1, feat2, feat3, code


class TxtNet_T(nn.Module):
    """文本编码网络backbone"""
    def __init__(self, text_length=512, code_len=64):
        super(TxtNet_T, self).__init__()
        self.txt_encoder1 = nn.Sequential(
            nn.Linear(text_length, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.txt_encoder2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.txt_encoder3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.txtHashing = nn.Sequential(
            nn.Linear(512, code_len),
        )

    def forward(self, x):
        feat1 = self.txt_encoder1(x)
        feat2 = self.txt_encoder2(feat1)
        feat3 = self.txt_encoder3(feat2)
        code = self.txtHashing(feat3)
        code = torch.tanh(code)
        return feat1, feat2, feat3, code
