import torch
import torch.nn as nn
import torch.nn.functional as F

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