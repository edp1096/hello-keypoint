from consts import device
from config import *

import torch.nn as nn
from torchvision import datasets, models, transforms
import timm

from modules.loss import ArcFace


class NetHead(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(NetHead, self).__init__()

        in_features = None
        weights = None

        self.embedding = None  # features embedding

        # match MODEL_NAME:
        if MODEL_NAME == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
            in_features, self.model.fc = self.model.fc.in_features, nn.Identity()
        if MODEL_NAME == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            in_features, self.model.fc = self.model.fc.in_features, nn.Identity()
        elif MODEL_NAME == "efficientnetv2_s":
            model_name = "tf_efficientnetv2_s_in21k"
            self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, drop_rate=0.2, drop_path_rate=0.2)
            in_features, self.model.classifier = self.model.classifier.in_features, nn.Identity()

        self.fc = nn.Linear(in_features, num_classes)

    def getEmbeddingHook(self, net_model, input, output):
        self.embedding = output

    def forward(self, x, label=None):
        x = self.model(x.float())
        x = self.fc(x)

        return self.embedding, x
