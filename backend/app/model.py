import torch
import torch.nn as nn

class PitchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PitchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어 (시계열 패턴 분석)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Fully Connected 레이어 (결과 분류)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 초기 은닉 상태와 셀 상태 0으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시퀀스의 출력만 사용하여 분류 (Many-to-One)
        out = self.fc(out[:, -1, :])
        return out