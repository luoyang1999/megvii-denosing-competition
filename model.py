import numbers

import megengine.random
import megengine.module as M
import megengine.functional as F
from megengine import Parameter

from pixshuffle import pixel_shuffle, pixel_shuffle_inv


def to_3d(x):
    batch, channel, height, width = x.shape
    x = F.transpose(x, (0, 2, 3, 1))
    x = F.reshape(x, (batch, height * width, channel))
    return x

def to_4d(x, h, w):
    batch, piexl, channel = x.shape
    x = F.transpose(x, (0, 2, 1))
    x = F.reshape(x, (batch, channel, h, w))
    return x

def to_multi_head(x, head):
    batch, channel, height, width = x.shape
    x = F.reshape(x, (batch, head, channel // head, height * width))
    return x

class BiasFree_LayerNorm(M.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = normalized_shape

        assert len(normalized_shape) == 1

        self.weight = Parameter(F.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdims=True, unbiased=False)
        return x / F.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(M.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = normalized_shape

        assert len(normalized_shape) == 1

        self.weight = Parameter(F.ones(normalized_shape))
        self.bias = Parameter(F.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = F.mean(x, axis=-1, keepdims=True)
        sigma = F.var(x, axis=-1, keepdims=True)
        return (x - mu) / F.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(M.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(M.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = M.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = M.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                               groups=hidden_features * 2, bias=bias)

        self.project_out = M.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = F.split(x, nsplits_or_sections=2, axis=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(M.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = Parameter(F.ones((num_heads, 1, 1)))

        self.qkv = M.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = M.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = M.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = F.split(qkv, 3, axis=1)

        q = to_multi_head(q, self.num_heads)
        k = to_multi_head(k, self.num_heads)
        v = to_multi_head(v, self.num_heads)

        q = F.normalize(q, axis=-1)
        k = F.normalize(k, axis=-1)

        attn = F.matmul(q, k.transpose((0, 1, 3, 2))) * self.temperature
        attn = F.softmax(attn, axis=-1)

        out = F.matmul(attn, v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        batch, head, channel_per_head, pixel = out.shape
        out = F.reshape(out, (batch, self.num_heads * channel_per_head, h, w))

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(M.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(M.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = M.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(M.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.conv = M.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # [B, C, H, W] -> [B, C/2, H, W]
        x = self.conv(x)
        # [B, C/2, H, W] -> [B, 2C, H/2, W/2]
        x = pixel_shuffle_inv(x, 2)
        return x


class Upsample(M.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.conv = M.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False, groups=4)

    def forward(self, x):
        # [B, C, H, W] -> [B, 2C, H, W]
        x = self.conv(x)
        # [B, 2C, H, W] -> [B, C/2, 2H, 2W]
        x = pixel_shuffle(x, 2)
        return x


class Bili_resize(M.Module):
    def __init__(self, factor):
        super(Bili_resize, self).__init__()
        self.factor = factor

    def forward(self, x):
        x = F.vision.interpolate(x, scale_factor=self.factor)
        return x


##########################################################################
## SKFF modules
class SKFF(M.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.conv_du = M.Sequential(M.Conv2d(in_channels, d, 1, padding=0, bias=bias), M.PReLU())

        self.fcs = []
        for i in range(self.height):
            self.fcs.append(M.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = M.Softmax(axis=1)

    def forward(self, inp_feats):
        # 取其中一层，看输入的shape [batch, feature, h, w]
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = F.concat(inp_feats, axis=1)
        # [batch, height, feature, h, w]
        inp_feats = inp_feats.reshape(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        # [batch, height, feature, h, w] -> [batch, feature, h, w]
        feats_U = F.sum(inp_feats, axis=1)
        # [batch, feaure, h, w] -> [batch, feature, 1, 1]
        feats_S = self.avg_pool(feats_U)
        # [batch, feaure, 1, 1] -> [batch, feature / reduction, 1, 1]
        feats_Z = self.conv_du(feats_S)
        # [batch, feature / reduction, 1, 1] -> height * [batch, feature, 1, 1] 采用height个卷积核分别卷积
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # height * [batch, feature, 1, 1] -> [batch, height*feature, 1, 1]
        attention_vectors = F.concat(attention_vectors, axis=1)
        # [batch, height*feature, 1, 1] -> [batch, height, feature, 1, 1]
        attention_vectors = attention_vectors.reshape(batch_size, self.height, n_feats, 1, 1)
        # [batch, height, feature, 1, 1] shape不变，softmax看哪个height更重要
        attention_vectors = self.softmax(attention_vectors)
        # [batch, height, feature, h, w] * [batch, height, feature, 1, 1] = [batch, height, feature, h, w]
        # [batch, height, feature, h, w] -> [batch, feature, h, w]
        feats_V = F.sum(inp_feats * attention_vectors, axis=1)

        return feats_V

##########################################################################
## ConvBlock

class ConvBlock(M.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(ConvBlock, self).__init__()
        self.block = M.Sequential(
            M.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            M.LeakyReLU(relu_slope),
            M.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            M.LeakyReLU(relu_slope)
        )

        self.shortcut = M.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.block(x)
        return x + sc

##########################################################################
## SSA modules
class SSA(M.Module):
    def __init__(self, in_size, subspace_dim=4, relu_slope=0.2):
        super(SSA, self).__init__()
        self.conv_block = ConvBlock(in_size, subspace_dim, relu_slope)
        self.num_subspace = subspace_dim

    def forward(self, up, bridge):
        b_, c_, h_, w_ = bridge.shape
        out = F.concat([up, bridge], 1)

        sub = self.conv_block(out)
        V_t = sub.reshape(b_, self.num_subspace, h_*w_)
        # [batchsize, k, h*w]
        V_t = V_t / (1e-6 + F.abs(V_t).sum(axis=2, keepdims=True))
        # [batchsize, h*w, k]
        V = V_t.transpose(0, 2, 1)
        # [batchsize, k, h*w] * [batchsize, h*w, k] -> [batchsize, k, k]
        mat = F.matmul(V_t, V)
        # 求逆矩阵
        mat_inv = F.matinv(mat)
        # [batchsize, k, k] * [batchsize, k, h*w] -> [batchsize, k, h*w]
        project_mat = F.matmul(mat_inv, V_t)
        bridge_ = bridge.reshape(b_, c_, h_*w_)
        # [batchsize, k, h*w] * [batchsize, h*w, c] -> [batchsize, k, c]
        project_feature = F.matmul(project_mat, bridge_.transpose(0, 2, 1))
        # [batchsize, h*w, k] * [batchsize, k, c] -> [batchsize, c, h*w] -> [batchsize, c, h, w]
        bridge = F.matmul(V, project_feature).transpose(0, 2, 1).reshape(b_, c_, h_, w_)

        return bridge


##########################################################################
##---------- Restormer -----------------------
class Restormer_skffv3_ssa_share(M.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=8,
                 num_blocks=[2, 3, 3],
                 num_refinement_blocks=3,
                 shared_num=2,              # 权重共享重复次数
                 heads=[2, 4, 8],
                 ffn_expansion_factor=2.18,  # 2.22
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree' 'WithBias'
                 ):
        super(Restormer_skffv3_ssa_share, self).__init__()
        # transformer block堆叠权重共享次数
        self.shared_num = shared_num

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.skff_down1 = M.Sequential(M.Conv2d(dim, 2*dim, 3, 1, 1, groups=dim), Bili_resize(0.5))

        self.encoder_level1 = M.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(int(dim * 2 ** 1))  ## From Level 1 to Level 2
        self.encoder_level2 = M.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 4 ** 1))  ## From Level 2 to Level 3

        self.latent = M.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 增加上采样，将其搞成256进行注意力机制
        self.skff_up1 = M.Sequential(M.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), 3, stride=1, padding=1),
                                     Bili_resize(4))

        # 增加ssa模块
        self.ssa1 = SSA(in_size=int(dim * 2 ** 2))

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = M.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = M.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        # 增加上采样，将其搞成256进行注意力机制
        self.skff_up2 = M.Sequential(Bili_resize(2))
        # 增加ssa模块
        self.ssa2 = SSA(in_size=int(dim * 2 ** 1))

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = M.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                              LayerNorm_type=LayerNorm_type) for i in
                                             range(num_blocks[0])])

        # 三个上采样输入skff模块
        self.skff = SKFF(in_channels=int(dim * 2 ** 1), height=3)

        self.refinement = M.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = M.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # [batch_size, channels=1, height, width] -> [batch_size, dim, height, width]
        inp_enc_level1 = self.patch_embed(inp_img)
        skff_down1 = self.skff_down1(inp_enc_level1)

        # [batch_size, dim, height, width] 维度不变
        inp = megengine.Tensor(inp_enc_level1)
        for i in range(self.shared_num):
            inp = self.encoder_level1(inp)
        out_enc_level1 = inp
        # [batch_size, dim, height, width] -> [batch_size, 2*dim, height//2, width//2]
        inp_enc_level2 = self.down1_2(F.concat([out_enc_level1, inp_enc_level1], axis=1))

        # [batch_size, 2*dim, height//2, width//2] 维度不变
        inp = megengine.Tensor(inp_enc_level2)
        for i in range(self.shared_num):
            inp = self.encoder_level2(inp)
        out_enc_level2 = inp
        # [batch_size, 2*dim, height//2, width//2] -> [batch_size, 4*dim, height//4, width//4]
        inp_enc_level3 = self.down2_3(F.concat([out_enc_level2, skff_down1], axis=1))

        # [batch_size, 4*dim, height//4, width//4] 维度不变
        inp = megengine.Tensor(inp_enc_level3)
        for i in range(self.shared_num):
            inp = self.latent(inp)
        latent = inp

        # 上采样一个 [batch_size, 4*dim, height//4, width//4] -> [batch_size, 2*dim, height, width]
        skff_up1 = self.skff_up1(latent)

        # [batch_size, 4*dim, height//4, width//4] -> [batch_size, 2*dim, height//2, width//2]
        inp_dec_level2 = self.up3_2(latent)  # out_dec_level3

        # [batch_size, 2*dim, height//2, width//2] -> [batch_size, 2*dim, height//2, width//2]
        ssa1_out = self.ssa1(inp_dec_level2, out_enc_level2)
        # [batch_size, 2*dim, height//2, width//2] concat [batch_size, 2*dim, height//2, width//2] = [batch_size, 4*dim, height//2, width//2]
        inp_dec_level2 = F.concat([inp_dec_level2, ssa1_out], axis=1)
        # [batch_size, 4*dim, height//2, width//2] -> [batch_size, 2*dim, height//2, width//2]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # [batch_size, 2*dim, height//2, width//2] 维度不变
        inp = megengine.Tensor(inp_dec_level2)
        for i in range(self.shared_num):
            inp = self.decoder_level2(inp)
        out_dec_level2 = inp

        # 上采样一个 [batch_size, 2*dim, height//2, width//2] -> [batch_size, 2*dim, height, width]
        skff_up2 = self.skff_up2(out_dec_level2)

        # [batch_size, 2*dim, height//2, width//2] -> [batch_size, dim, height, width]
        inp_dec_level1 = self.up2_1(out_dec_level2)
        # [batch_size, dim, height, width] -> [batch_size, dim, height, width]
        ssa2_out = self.ssa2(inp_dec_level1, out_enc_level1)
        # [batch_size, dim, height, width] concat [batch_size, dim, height, width] = [batch_size, 2*dim, height, width]
        inp_dec_level1 = F.concat([inp_dec_level1, ssa2_out], axis=1)
        # [batch_size, 2*dim, height, width] 维度不变
        inp = megengine.Tensor(inp_dec_level1)
        for i in range(self.shared_num):
            inp = self.decoder_level1(inp)
        out_dec_level1 = inp

        # 3个上采样输入skff中  3*[batch_size, 2*dim, height, width] -> [batch_size, 2*dim, height, width]
        final_skff = self.skff([skff_up1, skff_up2, out_dec_level1])

        # [batch_size, 2*dim, height, width] 维度不变
        inp = megengine.Tensor(final_skff+out_dec_level1)
        for i in range(self.shared_num):
            inp = self.refinement(inp)
        out_refine = inp

        # [batch_size, 2*dim, height, width] -> [batch_size, channels=1, height, width] + [batch_size, channels=1, height, width]
        out_img = self.output(out_refine) + inp_img

        return out_img


if __name__ == '__main__':
    restormer = Restormer_skffv3_ssa_share(shared_num=3)
    # Parameters: 99405
    print('Parameters:', sum([i.size for i in restormer.parameters()]))
