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
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w4 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b4 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
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
            torch.einsum('bli,ii->bli', x.real, y[0]) -
            torch.einsum('bli,ii->bli', x.imag, y[1]) +
            b[0]
        )
        # 虚部
        imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, y[0]) +
            torch.einsum('bli,ii->bli', x.real, y[1]) +
            b[1]
        )

        # 1 layer
        result = torch.stack([real, imag], dim=-1)
        result = F.softshrink(result, lambd=self.sparsity_threshold)
        result = torch.view_as_complex(result)

        return result

    # FourierGNN 深度为3
    # def fourierGC(self, x, B, N, L):
    #    o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
    #                          device=x.device)
    #    o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
    #                          device=x.device)
    #    o2_real = torch.zeros(x.shape, device=x.device)
    #    o2_imag = torch.zeros(x.shape, device=x.device)
#
    #    o3_real = torch.zeros(x.shape, device=x.device)
    #    o3_imag = torch.zeros(x.shape, device=x.device)
#
    #    # 实部 'bli,ii->bli' 意思是 [b,l,i] * [i,i] -> [b,l,i]
    #    # x = a + bi ; y = c + di 则有 xy = ac - bd + (ad + bc)i
    #    # w也是一个复数，w[0]是实部，w[1]是虚部
#
    #    o1_real = F.relu(
    #        torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
    #        torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
    #        self.b1[0]
    #    )
    #    # 虚部
    #    o1_imag = F.relu(
    #        torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
    #        torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
    #        self.b1[1]
    #    )
#
    #    # 1 layer
    #    y = torch.stack([o1_real, o1_imag], dim=-1)
    #    y = F.softshrink(y, lambd=self.sparsity_threshold)
#
    #    o2_real = F.relu(
    #        torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
    #        torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
    #        self.b2[0]
    #    )
#
    #    o2_imag = F.relu(
    #        torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
    #        torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
    #        self.b2[1]
    #    )
#
    #    # 2 layer
    #    x = torch.stack([o2_real, o2_imag], dim=-1)
    #    x = F.softshrink(x, lambd=self.sparsity_threshold)
    #    x = x + y
#
    #    o3_real = F.relu(
    #            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
    #            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
    #            self.b3[0]
    #    )
#
    #    o3_imag = F.relu(
    #            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
    #            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
    #            self.b3[1]
    #    )
#
    #    # 3 layer
    #    z = torch.stack([o3_real, o3_imag], dim=-1)
    #    z = F.softshrink(z, lambd=self.sparsity_threshold)
    #    z = z + x
    #    z = torch.view_as_complex(z)
    #    return z

    def fourierGC(self, x0, B, N, L):

        o1 = self.complex_multiplication(x0, self.w1, self.b1)
        o2 = self.complex_multiplication(o1, self.w2, self.b2)
        o3 = self.complex_multiplication(o2, self.w3, self.b3)
        o4 = self.complex_multiplication(o3, self.w4, self.b4)

        op = o1+o2+o3+o4

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

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection 一个可训练的映射self.seq_length(回溯序列长度) -> 8维 定长序列
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

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