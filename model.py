import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(0)
        x = x + weight
        return self.dropout(x)


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        z = self.gelu(self.linear1(x))
        z = self.linear2(z)

        return x+z


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, adapter_size=64, dim_feedforward=3072, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.adapter1 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adapter2 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.dropout1(src2)
        src2 = self.adapter1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src2 = self.adapter2(src2)
        src = self.norm2(src + src2)

        return src

    def activate_adapter(self):
        tune_layers = [self.adapter1, self.adapter2, self.norm1, self.norm2]
        for layer in tune_layers:
            for param in layer.parameters():
                param.requires_grad = True


class ParamEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # BERT输出维度(768)到模型维度的映射
        self.char_embedding = nn.Linear(768, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, param_vectors):
        """
        Args:
            param_vectors: 列表的列表，每个内部列表包含该位置所有参数的向量
                         shape: [batch_size, seq_len, num_params, 768]
        Returns:
            encoded: shape [batch_size, seq_len, d_model]
        """
        # 对每个参数向量进行线性映射
        encoded = []
        for batch in param_vectors:
            batch_encoded = []
            for seq_params in batch:
                if len(seq_params) == 0:  # 处理空序列
                    # 使用零向量
                    seq_encoding = torch.zeros(self.d_model, 
                                            device=self.char_embedding.weight.device)
                else:
                    # 将所有参数向量映射到相同维度并求平均
                    param_embeddings = [self.char_embedding(p) for p in seq_params]
                    seq_encoding = torch.stack(param_embeddings).mean(dim=0)
                batch_encoded.append(seq_encoding)
            encoded.append(torch.stack(batch_encoded))
        
        encoded = torch.stack(encoded)
        return self.norm(encoded)


class LogAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 参数编码器
        self.param_encoder = ParamEncoder(d_model)
        
        # 多头注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, 
            batch_first=batch_first, **factory_kwargs)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self.nhead = nhead
        self.d_model = d_model
        self.batch_first = batch_first

    def forward(self, src, param_vectors, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: 模板序列 [batch_size, seq_len, d_model]
            param_vectors: 参数向量列表的列表 [batch_size, seq_len, num_params, 768]
        """
        batch_size = src.size(0)
        seq_len = src.size(1)
        head_dim = self.d_model // self.nhead
        
        # 1. 参数编码：将每个位置的所有参数编码成一个向量
        param_encoding = self.param_encoder(param_vectors)  # [batch_size, seq_len, d_model]
        
        # 2. 计算注意力偏置
        param_bias = param_encoding.view(batch_size, seq_len, self.nhead, head_dim)
        param_bias = param_bias.permute(0, 2, 1, 3)  # [batch_size, nhead, seq_len, head_dim]
        
        # 计算参数注意力偏置
        param_bias = torch.matmul(param_bias, param_bias.transpose(-2, -1))
        param_bias = param_bias / math.sqrt(head_dim)  # [batch_size, nhead, seq_len, seq_len]
        
        # 3. 应用带参数偏置的注意力
        # 注：直接使用nn.MultiheadAttention无法添加偏置，需要自己实现注意力计算
        q = src.view(batch_size, seq_len, self.nhead, head_dim).permute(0, 2, 1, 3)
        k = src.view(batch_size, seq_len, self.nhead, head_dim).permute(0, 2, 1, 3)
        v = src.view(batch_size, seq_len, self.nhead, head_dim).permute(0, 2, 1, 3)
        
        # QK^T + 参数偏置
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = attn + param_bias  # 添加参数偏置
        
        # 应用注意力mask（如果有）
        if src_mask is not None:
            attn = attn.masked_fill(src_mask == 0, float('-inf'))
        
        # Softmax和Dropout
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout1.p, training=self.training)
        
        # 应用注意力权重
        output = torch.matmul(attn, v)  # [batch_size, nhead, seq_len, head_dim]
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, nhead, head_dim]
        output = output.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # 残差连接和层归一化
        src = src + self.dropout1(output)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class Model(nn.Module):
    def __init__(self, mode, num_layers=4, adapter_size=64, dim=768, window_size=100, 
                 nhead=8, dim_feedforward=3072, dropout=0.1):
        super(Model, self).__init__()
        
        encoder_layer = LogAttentionTransformerLayer(
            dim, nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True)
        
        self.transformer_encoder = nn.ModuleList(
            [encoder_layer for _ in range(num_layers)])
            
        self.pos_encoder = PositionalEncoding(d_model=dim)
        
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x):
        """
        Args:
            x: (template_seq, param_seq) 元组
                template_seq: [batch_size, seq_len, dim]
                param_seq: list[batch_size][seq_len][num_params, 768] - 参数列表
        """
        template_seq, param_seq = x
        
        # 只对模板序列应用位置编码
        template_seq = self.pos_encoder(template_seq)
        
        # 参数序列直接传给transformer层
        for layer in self.transformer_encoder:
            template_seq = layer(template_seq, param_seq)
        
        x = template_seq.contiguous().view(template_seq.size(0), -1)
        x = self.fc1(x)
        
        return x

    def train_adapter(self):
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.transformer_encoder:
            layer.activate_adapter()
        for param in self.fc1.parameters():
            param.requires_grad = True

    def train_classifier(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.fc1.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    model = Model('adapter')
    pass
