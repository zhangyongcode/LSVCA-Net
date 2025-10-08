# A lightweight scale-view co-awareness for efficient 3D tumor segmentation
from typing import Tuple
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from mamba_ssm import Mamba
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.utils import get_act_layer, get_norm_layer
#
import torch
from typing import Union
import torch.nn as nn
from monai.utils import InterpolateMode, UpsampleMode


def get_upsample_layer(in_ch: int, out_ch: int, upsample_mode: Union[UpsampleMode, str] = "nontrainable",
                       scale_factor: int = 2):
    return UpSample(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 ksize: int = 3,
                 dilation: int = 1,
                 stride: int = 1,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 spatial_dims=3
                 ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=out_ch, ksize=ksize, dilation=dilation, spatial_dims=spatial_dims,
                           stride=stride),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_ch),
            get_act_layer(act),
        )

    def forward(self, x):
        # print(x.size())
        x = self.conv(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class MVMamba(nn.Module):
    # Multi-View Mamba
    def __init__(self,
                 dim,
                 Mamba_view_flag=True,
                 spatial_dims: int = 3,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 ):
        super(MVMamba, self).__init__()
        self.dim = dim

        self.view1 = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=dim),
            get_act_layer(act),
        )

        self.view2 = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0)),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=dim),
            get_act_layer(act),
        )

        self.view3 = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=dim),
            get_act_layer(act),
        )

        self.mamba = MambaUnit(dim=self.dim)
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels=dim * 3, out_channels=dim, kernel_size=1, stride=1, padding=0),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=dim),
            get_act_layer(act),
        )
        self.view_flag = Mamba_view_flag

    def forward(self, x):
        if self.view_flag:
            view1 = self.mamba(self.view1(x))
            view2 = self.mamba(self.view2(x))
            view3 = self.mamba(self.view3(x))
            out = self.fusion(torch.cat([view1, view2, view3], dim=1))
        else:
            out = self.mamba(x)
        return out


class MambaUnit(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=1, expand=2, channel_token=False):
        super().__init__()
        # print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out


def get_conv_layer(in_ch: int, out_ch: int, ksize: int = 3, stride: int = 1, bias: bool = False, dilation=1,
                   spatial_dims=3, groups=1):
    return Convolution(spatial_dims, in_ch, out_ch, strides=stride, kernel_size=ksize, bias=bias, conv_only=True,
                       dilation=dilation, groups=groups)


class MVConv(nn.Module):
    # Multi-View Conv
    def __init__(self,
                 in_ch: int,
                 ksize: int = 3,
                 ksize2d: int = 9,
                 spatial_dims=3,
                 MVConv_flag=True,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize, groups=in_ch, padding=ksize // 2),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_ch),
            get_act_layer(act),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize2d, groups=in_ch,
                      padding=ksize2d // 2),
            get_norm_layer(name=norm, spatial_dims=2, channels=in_ch),
            nn.Sigmoid(),  # shared
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize2d, groups=in_ch,
                      padding=ksize2d // 2),
            get_norm_layer(name=norm, spatial_dims=2, channels=in_ch),
            nn.Sigmoid(),  # shared
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ksize2d, groups=in_ch,
                      padding=ksize2d // 2),
            get_norm_layer(name=norm, spatial_dims=2, channels=in_ch),
            nn.Sigmoid(),  # shared
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch * 4, out_channels=in_ch, kernel_size=1, ),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_ch),

        )
        self.act = get_act_layer(act)
        self.sigmoid = nn.Sigmoid()

        self.view_flag = MVConv_flag
        self.pwconv = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, ),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_ch),
            get_act_layer(act),  # 进行融合   Sag
        )

    def forward(self, x):
        # 全角度建模用于获取整体的初略感知
        if self.view_flag:
            conv1 = self.conv1(x)

            B, C, D, H, W = conv1.shape
            # W-D维度
            df_wd = F.avg_pool3d(conv1, [D, 1, 1])
            df_wd = torch.squeeze(df_wd, dim=2)
            df_wd = self.conv2_1(df_wd)
            df_wd = torch.unsqueeze(df_wd, dim=2)  # ->(B, C, 1, W, D)
            df_wd_out = df_wd * conv1

            df_hd = F.avg_pool3d(conv1, [1, H, 1])
            df_hd = torch.squeeze(df_hd, dim=3)
            df_hd = self.conv2_2(df_hd)
            df_hd = torch.unsqueeze(df_hd, dim=3)  # ->(B, C, H, 1, D)
            df_hd_out = df_hd * conv1

            df_hw = F.avg_pool3d(conv1, [1, 1, W])
            df_hw = torch.squeeze(df_hw, dim=4)
            df_hw = self.conv2_3(df_hw)
            df_hw = torch.unsqueeze(df_hw, dim=4)  # ->(B, C, H, 1, D)
            df_hw_out = df_hw * conv1

            conv2_out = torch.cat([df_wd_out, df_hd_out, df_hw_out, x], dim=1)
            # 通道modeling 来获取进一步的特征感知
            conv3_out = self.conv3(conv2_out)
            out = self.act(conv3_out + x)

        else:
            conv1 = self.conv1(x)
            out = self.pwconv(conv1)
        return out


class DownSample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True})) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            get_conv_layer(in_ch=in_ch, out_ch=in_ch, ksize=1, stride=2, groups=in_ch),
            get_norm_layer(name=norm, spatial_dims=3, channels=in_ch),
            get_act_layer(act),
            get_conv_layer(in_ch=in_ch, out_ch=out_ch, ksize=1, stride=1),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),
        )

    def forward(self, x):
        out = self.conv1(x)
        return out


class MVLGM(nn.Module):
    # multi-view local-global mix
    def __init__(self,
                 in_ch: int,
                 ksize2d: int = 9,
                 M: int = 4,
                 N: int = 4,
                 MVConv_flag=True,
                 Mamba_view_flag=True,
                 spatial_dims: int = 3,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = in_ch // 2
        self.local_branch = nn.Sequential(
            *[MVConv(self.dim, ksize2d=ksize2d, MVConv_flag=MVConv_flag) for _ in range(M)])
        self.global_branch = nn.Sequential(*[MVMamba(self.dim, Mamba_view_flag=Mamba_view_flag) for _ in range(N)])
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, ),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_ch),
            get_act_layer(act),  # 进行融合   Sag
        )

    def forward(self, x):
        x0, x1 = torch.split(x, self.dim, dim=1)
        x0_out = self.local_branch(x0)
        x1_out = self.global_branch(x1)
        fusion = torch.cat([x0_out, x1_out], dim=1)
        out = self.fusion(fusion)
        return out


class MVMSF(nn.Module):
    # Multi-View Multi-Stage Fusion
    def __init__(self, init_filters, out_ch,
                 norm: Union[Tuple, str] = ("instance", {"affine": True}),
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 view_flag: bool = True,
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE, ):
        super(MVMSF, self).__init__()

        self.compress_16X = nn.Sequential(ConvBlock(init_filters * 4, out_ch, ksize=1),
                                          get_upsample_layer(out_ch, out_ch=out_ch, upsample_mode=upsample_mode,
                                                             scale_factor=16)
                                          )
        self.compress_8X = nn.Sequential(ConvBlock(init_filters * 3, out_ch, ksize=1),
                                         get_upsample_layer(out_ch, out_ch=out_ch, upsample_mode=upsample_mode,
                                                            scale_factor=8)
                                         )
        self.compress_4X = nn.Sequential(ConvBlock(init_filters * 2, out_ch, ksize=1),
                                         get_upsample_layer(out_ch, out_ch=out_ch, upsample_mode=upsample_mode,
                                                            scale_factor=4)
                                         )
        self.compress_2X = nn.Sequential(ConvBlock(init_filters * 1, out_ch, ksize=1),
                                         get_upsample_layer(out_ch, out_ch=out_ch, upsample_mode=upsample_mode,
                                                            scale_factor=2)
                                         )

        self.fusion = ConvBlock(out_ch * 4, out_ch, ksize=1, )
        asize = 3
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=asize, groups=out_ch, padding=asize // 2),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, asize, asize), groups=out_ch,
                      padding=(0, asize // 2, asize // 2)),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),  # 不同角度的扫描感知   axis
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(asize, 1, asize), groups=out_ch,
                      padding=(asize // 2, 0, asize // 2)),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),  # 不同角度的扫描感知   Cor
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(asize, asize, 1), groups=out_ch,
                      padding=(asize // 2, asize // 2, 0)),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),  # 不同角度的扫描感知   Sag
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=out_ch * 3, out_channels=out_ch, kernel_size=1, ),
            get_norm_layer(name=norm, spatial_dims=3, channels=out_ch),
            get_act_layer(act),  # 进行融合   Sag
        )
        self.flag = view_flag
        # self.act = get_act_layer(act)

    def forward(self, E1, E2, E3, E4):
        # E1=(B, C, D/2, H/2, W/2), E2=(B, C,D/4, H/4, W/4), E3=(B, C,D/8, H/8, W/8), E1=(B, C, D/16, H/16, W/16)
        E4_up = self.compress_16X(E4)
        E3_up = self.compress_8X(E3)
        E2_up = self.compress_4X(E2)
        E1_up = self.compress_2X(E1)
        x = self.fusion(torch.cat([E1_up, E2_up, E3_up, E4_up], dim=1))
        # 实现角度的多尺度
        if self.flag:
            conv1 = self.conv1(x)
            conv2_1_out = self.conv2_1(conv1)
            conv2_2_out = self.conv2_2(conv1 + conv2_1_out)
            conv2_3_out = self.conv2_3(conv1 + conv2_2_out)
            conv2_out = torch.cat([conv2_1_out, conv2_2_out, conv2_3_out], dim=1)
            x = self.conv3(conv2_out) + x
        return x


class LSVCANet(nn.Module):
    # A Lightweight Scale-View Co-Awareness Network

    def __init__(self,
                 in_ch: int = 4,
                 out_ch: int = 3,
                 init_filters: int = 16,
                 ksize2d: int = 9,
                 N: int = 4,
                 M: int = 4,
                 view_flag: bool = True,
                 MVConv_flag=True,
                 Mamba_view_flag=True,
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder1 = nn.Sequential(
            ConvBlock(in_ch=in_ch, out_ch=init_filters * 1, stride=2),
            ConvBlock(in_ch=init_filters * 1, out_ch=init_filters * 1)
        )  # 1/2
        self.encoder2 = nn.Sequential(
            DownSample(in_ch=init_filters * 1, out_ch=init_filters * 2),
            ConvBlock(in_ch=init_filters * 2, out_ch=init_filters * 2)
        )  # 1/4
        self.encoder3 = nn.Sequential(
            DownSample(in_ch=init_filters * 2, out_ch=init_filters * 3),
            MVLGM(in_ch=init_filters * 3, MVConv_flag=MVConv_flag, Mamba_view_flag=Mamba_view_flag, ksize2d=ksize2d,
                  N=N, M=M),

        )  # 1/8

        self.encoder4 = nn.Sequential(
            DownSample(in_ch=init_filters * 3, out_ch=init_filters * 4),
            MVLGM(in_ch=init_filters * 4, MVConv_flag=MVConv_flag, Mamba_view_flag=Mamba_view_flag, ksize2d=ksize2d,
                  N=N, M=M),

        )  # 1/8

        self.fusion = MVMSF(init_filters=init_filters, out_ch=init_filters // 2, upsample_mode=upsample_mode,
                            view_flag=view_flag)
        self.conv_final = get_conv_layer(in_ch=init_filters // 2, out_ch=out_ch, ksize=3)

    def forward(self, x):
        # encoder process
        encoder1 = self.encoder1(x)  # 1/2
        encoder2 = self.encoder2(encoder1)  # 1/4
        encoder3 = self.encoder3(encoder2)  # 1/8
        encoder4 = self.encoder4(encoder3)  # 1/16

        fusion = self.fusion(encoder1, encoder2, encoder3, encoder4)
        out = self.conv_final(fusion)  #
        return out


if __name__ == '__main__':
    import torch
    from thop import profile, clever_format

    # for Brain tumor segmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LSVCANet(in_ch=4, init_filters=24, ksize2d=9, out_ch=3,
                         MVConv_flag=True, Mamba_view_flag=True, view_flag=True, N=4, M=4)
    model = net.to(device)
    model.train()
    input_tensor = torch.randn(1, 4, 128, 128, 128).to(device)  # 192 FLOPs: 38.845G params: 381.600K
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("Brain Tumor Segmentation Task" "params: %s" % (params), "FLOPs: %s" % (flops))

    # For FLARE dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LSVCANet(in_ch=1, init_filters=24, ksize2d=9, out_ch=5,
                         MVConv_flag=True, Mamba_view_flag=True, view_flag=True, N=4, M=4)
    model = net.to(device)
    model.train()
    input_tensor = torch.randn(1, 1, 192, 192, 192).to(device)  # 192 FLOPs: 38.845G params: 381.600K
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLARE Segmentation Task" "params: %s" % (params), "FLOPs: %s" % (flops))
