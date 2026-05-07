"""
Week 8: LSTM with Multi-head Self-Attention for Pitch Sequencing

Transformer 기반 Attention 메커니즘을 적용한 투구 시퀀스 모델링

수학적 배경:
-----------
1. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   
2. Multi-head Attention:
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
   
3. Positional Encoding:
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

참고: Vaswani et al. (2017). "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# Part 1: Original Bahdanau Attention (Legacy, Week 4)
# ═══════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """
    Bahdanau Attention 스타일의 메커니즘
    LSTM의 모든 시점(Time steps)의 출력을 참고하여 
    어떤 시점이 예측에 중요한지 가중치(Attention Weights)를 계산함
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # 어텐션 스코어 계산을 위한 레이어
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        
        # 1. 각 시점(time step)별 점수 계산
        # energy shape: (batch_size, seq_len, 1)
        energy = torch.tanh(self.attn(lstm_output)) 
        
        # 2. 확률 분포로 변환 (Softmax) -> 이것이 바로 '가중치'
        # weights shape: (batch_size, seq_len, 1)
        weights = F.softmax(energy, dim=1)
        
        # 3. 가중치를 적용한 가중합(Context Vector) 계산
        # context_vector shape: (batch_size, hidden_size)
        # (batch, hidden, seq) @ (batch, seq, 1) -> (batch, hidden, 1)
        context_vector = torch.bmm(lstm_output.transpose(1, 2), weights).squeeze(2)
        
        return context_vector, weights

class PitchLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PitchLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM 레이어 (return_sequences=True와 유사하게 모든 출력 반환)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 2. 어텐션 레이어 추가
        self.attention = Attention(hidden_size)
        
        # 3. 분류기 (Context Vector를 입력으로 받음)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM 순전파 (모든 시점의 출력 output을 사용)
        # output shape: (batch, seq_len, hidden_size)
        output, (hn, cn) = self.lstm(x)
        
        # 어텐션 적용
        # context shape: (batch, hidden_size)
        # attn_weights: 시각화에 사용할 수 있는 가중치
        context, attn_weights = self.attention(output)
        
        # 분류 수행
        out = self.dropout(context)
        out = self.fc(out)
        
        return out, attn_weights


# ═══════════════════════════════════════════════════════════════
# Part 2: Multi-head Self-Attention (Week 8)
# ═══════════════════════════════════════════════════════════════

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (Vaswani et al., 2017)
    
    수식:
    ----
    Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    
    Args:
    -----
    dropout: Attention weights dropout 비율
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        -----
        query: (batch_size, n_heads, seq_len, d_k)
        key: (batch_size, n_heads, seq_len, d_k)
        value: (batch_size, n_heads, seq_len, d_v)
        mask: (batch_size, 1, 1, seq_len) - Optional padding mask
        
        Returns:
        --------
        output: (batch_size, n_heads, seq_len, d_v)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-Attention
    
    여러 개의 attention head를 병렬로 실행하여 다양한 관점에서 시퀀스를 분석
    
    Args:
    -----
    d_model: Input/output dimension
    n_heads: Number of attention heads
    dropout: Dropout 비율
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        -----
        query: (batch_size, seq_len, d_model)
        key: (batch_size, seq_len, d_model)
        value: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, seq_len) - Optional
        
        Returns:
        --------
        output: (batch_size, seq_len, d_model)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Split into n_heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional Encoding (Sinusoidal)
    
    시퀀스 내 위치 정보를 임베딩에 추가
    
    Args:
    -----
    d_model: Embedding dimension
    max_len: Maximum sequence length
    dropout: Dropout 비율
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
        -----
        x: (batch_size, seq_len, d_model)
        
        Returns:
        --------
        (batch_size, seq_len, d_model) with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    수식: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    구조:
    1. Multi-head Self-Attention
    2. Add & Norm
    3. Feed-Forward
    4. Add & Norm
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention + Residual + Norm
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention_weights


class PitchTransformerModel(nn.Module):
    """
    🔥 [WEEK 8] LSTM + Multi-head Self-Attention for Pitch Prediction
    
    아키텍처:
    --------
    1. Input Embedding
    2. Positional Encoding
    3. Bi-LSTM
    4. Transformer Encoder (N layers)
    5. Global Average Pooling
    6. Classification Head
    
    Args:
    -----
    input_size: Number of features (39)
    d_model: Model dimension (128)
    n_heads: Number of attention heads (4)
    n_layers: Number of encoder layers (2)
    d_ff: Feed-forward hidden dim (512)
    num_classes: Number of pitch types (4)
    max_seq_len: Max sequence length (50)
    dropout: Dropout rate (0.1)
    """
    
    def __init__(self,
                 input_size: int = 39,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 512,
                 num_classes: int = 4,
                 max_seq_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
        -----
        x: (batch, seq_len, input_size) or (batch, input_size)
        mask: (batch, seq_len) - Optional
        
        Returns:
        --------
        logits: (batch, num_classes)
        attention_info: Dict with attention weights
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 1. Project input
        x = self.input_projection(x)
        
        # 2. Positional encoding
        x = self.pos_encoder(x)
        
        # 3. LSTM
        lstm_out, _ = self.lstm(x)
        
        # 4. Transformer encoder
        attention_weights = {}
        for i, layer in enumerate(self.encoder_layers):
            lstm_out, attn_weights = layer(lstm_out, mask)
            attention_weights[f'layer_{i}'] = attn_weights
        
        # 5. Global average pooling
        pooled = lstm_out.mean(dim=1)
        
        # 6. Classification
        logits = self.classifier(pooled)
        
        return logits, attention_weights


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def print_model_info(model: nn.Module):
    """모델 구조 및 파라미터 정보 출력"""
    print("=" * 70)
    print("Model Architecture")
    print("=" * 70)
    print(model)
    
    print("\n" + "=" * 70)
    print("Model Parameters")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("\nParameters by Layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:50s} {param.numel():>10,} ({list(param.shape)})")


# ═══════════════════════════════════════════════════════════════
# Test Code
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Testing LSTM + Multi-head Self-Attention Model")
    print("=" * 70)
    
    # Hyperparameters
    INPUT_SIZE = 39
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 512
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    SEQ_LEN = 10
    
    # Create model
    model = PitchTransformerModel(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        num_classes=NUM_CLASSES,
        max_seq_len=50,
        dropout=0.1
    )
    
    # Print info
    print_model_info(model)
    
    # Test forward
    print("\n" + "=" * 70)
    print("Test Forward Pass")
    print("=" * 70)
    
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    print(f"\nInput: {x.shape}")
    
    logits, attn_weights = model(x)
    
    print(f"Output: {logits.shape}")
    print(f"Attention layers: {len(attn_weights)}")
    
    for layer_name, weights in attn_weights.items():
        print(f"  {layer_name}: {weights.shape}")
    
    # Test single pitch
    print("\n" + "=" * 70)
    print("Test Single Pitch")
    print("=" * 70)
    
    x_single = torch.randn(BATCH_SIZE, INPUT_SIZE)
    logits_single, _ = model(x_single)
    print(f"Input: {x_single.shape}")
    print(f"Output: {logits_single.shape}")
    
    # Predictions
    print("\n" + "=" * 70)
    print("Test Predictions")
    print("=" * 70)
    
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    pitch_types = ['FF', 'SL', 'CH', 'CU']
    
    print("\nSample predictions (first 5):")
    for i in range(min(5, BATCH_SIZE)):
        print(f"  {i+1}. {pitch_types[preds[i]]}: {' '.join([f'{p}={probs[i,j]:.3f}' for j,p in enumerate(pitch_types)])}")
    
    print("\n" + "=" * 70)
    print("✅ All Tests Passed!")
    print("=" * 70)
