import torch.nn as nn
import torch.nn.functional as F
import PIL
import torch


class AFNet(nn.Module):
    def __init__(self):
        super(AFNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=740, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1,740)

        fc1_output = F.relu(self.fc1(conv5_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
        avg_pool = avg_pool.view(b, c)
        fc1 = self.fc1(avg_pool)
        fc1 = self.relu(fc1)
        fc2 = self.fc2(fc1)
        scale = self.sigmoid(fc2).view(b, c, 1, 1)
        return x * scale.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class AFNetWithResidualsAndAttention(nn.Module):
    def __init__(self):
        super(AFNetWithResidualsAndAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2, 1), padding=(2, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.channel_attention = ChannelAttention(20)  # 通道注意力模块
        self.spatial_attention = SpatialAttention()  # 空间注意力模块

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=780, out_features=10)  # 修改为正确的输入特征数
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

        # 定义适应维度的快捷连接
        self.shortcut1 = nn.Conv2d(1, 3, kernel_size=1, stride=(2, 1), bias=False)
        self.shortcut2 = nn.Conv2d(3, 5, kernel_size=1, stride=(2, 1), bias=False)
        self.shortcut3 = nn.Conv2d(5, 10, kernel_size=1, stride=(2, 1), bias=False)
        self.shortcut4 = nn.Conv2d(10, 20, kernel_size=1, stride=(2, 1), bias=False)

    def forward(self, input):
        x = input
        conv1_output = self.conv1(x)
        shortcut1_output = self.shortcut1(x)
        if conv1_output.size() != shortcut1_output.size():
            padding = (0, 0, conv1_output.size(2) - shortcut1_output.size(2), 0)
            shortcut1_output = F.pad(shortcut1_output, padding)
        x = conv1_output + shortcut1_output

        conv2_output = self.conv2(x)
        shortcut2_output = self.shortcut2(x)
        if conv2_output.size() != shortcut2_output.size():
            padding = (0, 0, conv2_output.size(2) - shortcut2_output.size(2), 0)
            shortcut2_output = F.pad(shortcut2_output, padding)
        x = conv2_output + shortcut2_output

        conv3_output = self.conv3(x)
        shortcut3_output = self.shortcut3(x)
        if conv3_output.size() != shortcut3_output.size():
            padding = (0, 0, conv3_output.size(2) - shortcut3_output.size(2), 0)
            shortcut3_output = F.pad(shortcut3_output, padding)
        x = conv3_output + shortcut3_output

        conv4_output = self.conv4(x)
        shortcut4_output = self.shortcut4(x)
        if conv4_output.size() != shortcut4_output.size():
            padding = (0, 0, conv4_output.size(2) - shortcut4_output.size(2), 0)
            shortcut4_output = F.pad(shortcut4_output, padding)
        x = conv4_output + shortcut4_output

        conv5_output = self.conv5(x)
        x = conv5_output

        # 应用通道注意力模块
        x = self.channel_attention(x)

        # 应用空间注意力模块
        spatial_attention_output = self.spatial_attention(x)
        x = x * spatial_attention_output

        x = x.view(x.size(0), -1)  # 修改为正确的调整形状方法

        fc1_output = F.relu(self.fc1(x))
        fc2_output = self.fc2(fc1_output)
        return fc2_output