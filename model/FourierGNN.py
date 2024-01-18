import torch
import torch.nn as nn
import torch.nn.functional as F


class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.08
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor, 2))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, 2))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size, 2))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size, 2))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor, 2))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, 2))
        self.w4 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor, 2))
        self.b4 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, 2))
        self.w5 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor, 2))
        self.b5 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, 2))
        self.w6 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor, 2))
        self.b6 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, 2))

        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))

        self.readout = nn.Sequential(
            nn.Linear(116, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    def complex_multiplication(self, x, y, b):
        """
        A complex multiplication

        Args:
            x: input1
            y: input2
            b: Bias

        Returns:
        x*y+b
        input1 * input2 + Bias
        """
        real = F.relu(
            torch.einsum('blim,iik->blik', x.real, y[0]) -
            torch.einsum('blim,iik->blik', x.imag, y[1]) +
            b[0]
        )
        # 虚部
        imag = F.relu(
            torch.einsum('blim,iik->blik', x.imag, y[0]) +
            torch.einsum('blim,iik->blik', x.real, y[1]) +
            b[1]
        )

        # 1 layer
        result = torch.stack([real, imag], dim=-1)
        result = F.softshrink(result, lambd=self.sparsity_threshold)
        result = torch.view_as_complex(result)
        return result


    def fourierGC(self, x0, B, N, L):

        o1 = self.complex_multiplication(x0, self.w1, self.b1)
        o2 = self.complex_multiplication(o1, self.w2, self.b2)
        o3 = self.complex_multiplication(o2, self.w3, self.b3)
        o4 = self.complex_multiplication(o3, self.w4, self.b4)
        o5 = self.complex_multiplication(o4, self.w5, self.b5)
        o6 = self.complex_multiplication(o5, self.w6, self.b6)

        op = o1+o2+o3+o4+o5+o6

        return op

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D 扩维 D是 嵌入向量的维度大小()
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D 快速傅里叶变换 T=
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        x = x.unsqueeze(3)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size, 2)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size, 2)
        x = x.permute(0, 1, 3, 4, 2)  # B, N, D, L

        # projection 一个可训练的映射self.seq_length(回溯序列长度) -> 8维 定长序列
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        # readout
        x = x.permute(0, 2, 1)
        x = self.readout(x)
        # 全连接
        x = self.fc1(x.squeeze())
        return x

## 可视化邻接矩阵
#import matplotlib.pyplot as plt
#import torch
#
## 创建一个二维矩阵`
#matrix = torch.einsum("bil,bjl->bij",x,x)
#matrix = torch.Tensor.cpu(matrix).detach().numpy()
#
## 绘制矩阵
#plt.imshow(matrix[0], cmap='viridis')
#
## 添加颜色条
#plt.colorbar()
#
## 显示图像
#plt.show()