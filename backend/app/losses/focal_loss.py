"""
Focal Loss for addressing class imbalance
Reference: Lin et al. (2017) - "Focal Loss for Dense Object Detection"

Focal Loss는 쉬운 샘플(잘 맞추는 샘플)의 가중치를 낮추고
어려운 샘플(틀리는 샘플)의 가중치를 높여서 학습합니다.

FL(pt) = -α(1-pt)^γ * log(pt)

여기서:
- pt: 정답 클래스의 예측 확률
- α (alpha): 클래스별 가중치 (희귀 클래스에 높은 가중치)
- γ (gamma): focusing parameter (보통 2.0)
  * γ=0: Cross Entropy와 동일
  * γ↑: 쉬운 샘플 가중치 더 많이 감소

MLB 투구 예측에서:
- Fastball (35%): 쉽게 맞춤 → 가중치 낮춤
- Knuckleball (0.04%): 어렵게 맞춤 → 가중치 높임
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    
    Args:
        alpha: 클래스별 가중치 텐서 (num_classes,) 또는 스칼라
               None이면 모든 클래스 동일 가중치
        gamma: focusing parameter (default: 2.0)
               높을수록 쉬운 샘플 가중치 감소
        reduction: 'mean', 'sum', 또는 'none'
        
    Examples:
        >>> # 균형 잡힌 가중치
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> 
        >>> # 클래스별 가중치 지정
        >>> alpha = torch.tensor([1.0, 2.0, 3.0])  # 희귀 클래스에 높은 가중치
        >>> loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) - raw logits (softmax 전)
            targets: (N,) - class indices
            
        Returns:
            loss: scalar (reduction='mean') or (N,) (reduction='none')
        """
        # Cross Entropy Loss 계산
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None
        )
        
        # pt (probability of true class) 계산
        # pt = exp(-CE) = P(true_class)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 계산
        # FL = (1 - pt)^γ * CE
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Reduction 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    구종별 빈도 기반 자동 가중치 계산 + Focal Loss
    
    Effective Number of Samples를 사용하여 클래스 가중치를 자동 계산합니다.
    Reference: Cui et al. (2019) - "Class-Balanced Loss Based on Effective Number of Samples"
    
    Effective Number: E_n = (1 - β^n) / (1 - β)
    여기서:
    - n: 클래스의 샘플 수
    - β: 재샘플링 확률 (보통 0.999 또는 0.9999)
    
    Args:
        class_counts: 각 클래스의 샘플 수 (numpy array or list)
        gamma: Focal Loss focusing parameter
        beta: Effective number 계산용 파라미터
        
    Examples:
        >>> # 구종별 개수
        >>> pitch_counts = np.array([35000, 18000, 11000, 9000, 8000, 7000, 3000, 2000, 40])
        >>> #                         FF     SL     CH     CU    SI    FC    FS    ST    KN
        >>> 
        >>> loss_fn = WeightedFocalLoss(class_counts=pitch_counts, gamma=2.0, beta=0.999)
        >>> 
        >>> # 학습
        >>> loss = loss_fn(logits, labels)
        >>> loss.backward()
    """
    
    def __init__(
        self, 
        class_counts: np.ndarray,
        gamma: float = 2.0,
        beta: float = 0.999,
        reduction: str = 'mean'
    ):
        super(WeightedFocalLoss, self).__init__()
        
        # Effective Number of Samples 계산
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        
        # 정규화 (합이 클래스 개수가 되도록)
        weights = weights / weights.sum() * len(weights)
        
        # 최소/최대 가중치 제한 (너무 극단적인 값 방지)
        weights = np.clip(weights, 0.1, 10.0)
        
        print(f"📊 Weighted Focal Loss Initialized:")
        print(f"   Beta: {beta}")
        print(f"   Gamma: {gamma}")
        print(f"   Class weights:")
        for i, (count, weight) in enumerate(zip(class_counts, weights)):
            print(f"      Class {i}: count={count:>6}, weight={weight:.3f}")
        
        # Tensor로 변환
        self.alpha = torch.FloatTensor(weights)
        self.focal = FocalLoss(alpha=self.alpha, gamma=gamma, reduction=reduction)
        self.class_counts = class_counts
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) - raw logits
            targets: (N,) - class indices
            
        Returns:
            loss: scalar
        """
        return self.focal(inputs, targets)
    
    def get_weights(self) -> torch.Tensor:
        """가중치 반환 (디버깅/분석용)"""
        return self.alpha
    
    def get_info(self) -> dict:
        """Loss 정보 반환"""
        return {
            'type': 'WeightedFocalLoss',
            'gamma': self.gamma,
            'beta': self.beta,
            'num_classes': len(self.class_counts),
            'class_counts': self.class_counts.tolist(),
            'weights': self.alpha.tolist()
        }


class AdaptiveFocalLoss(nn.Module):
    """
    학습 중 동적으로 gamma 조정하는 Adaptive Focal Loss
    
    초기: 높은 gamma로 어려운 샘플에 집중
    후반: 낮은 gamma로 전체적인 성능 향상
    
    Args:
        alpha: 클래스 가중치
        gamma_start: 초기 gamma
        gamma_end: 최종 gamma
        total_epochs: 전체 에폭 수
        
    Example:
        >>> loss_fn = AdaptiveFocalLoss(alpha=weights, gamma_start=3.0, gamma_end=1.0, total_epochs=50)
        >>> 
        >>> for epoch in range(50):
        ...     loss_fn.update_gamma(epoch)  # gamma 업데이트
        ...     # training...
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma_start: float = 3.0,
        gamma_end: float = 1.0,
        total_epochs: int = 50,
        reduction: str = 'mean'
    ):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.current_gamma = gamma_start
        self.reduction = reduction
        
        self.focal = FocalLoss(alpha=alpha, gamma=gamma_start, reduction=reduction)
    
    def update_gamma(self, epoch: int):
        """
        현재 에폭에 따라 gamma 업데이트
        선형 감소: gamma = gamma_start - (gamma_start - gamma_end) * (epoch / total_epochs)
        """
        progress = min(epoch / self.total_epochs, 1.0)
        self.current_gamma = self.gamma_start - (self.gamma_start - self.gamma_end) * progress
        self.focal.gamma = self.current_gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(inputs, targets)
    
    def get_current_gamma(self) -> float:
        """현재 gamma 값 반환"""
        return self.current_gamma


def test_focal_loss():
    """Focal Loss 테스트 함수"""
    print("🧪 Testing Focal Loss Implementation...")
    print("="*60)
    
    # 샘플 데이터
    batch_size = 100
    num_classes = 5
    
    # 불균형 클래스 분포 시뮬레이션
    # Class 0: 50%, Class 1: 25%, Class 2: 15%, Class 3: 8%, Class 4: 2%
    class_probs = torch.tensor([0.50, 0.25, 0.15, 0.08, 0.02])
    targets = torch.multinomial(class_probs, batch_size, replacement=True)
    
    # 랜덤 logits
    logits = torch.randn(batch_size, num_classes)
    
    # Test 1: Standard Focal Loss
    print("\nTest 1: Standard Focal Loss (gamma=2.0)")
    print("-"*60)
    focal_loss = FocalLoss(gamma=2.0)
    loss1 = focal_loss(logits, targets)
    print(f"✅ Loss: {loss1.item():.4f}")
    
    # Test 2: Focal Loss with alpha
    print("\nTest 2: Focal Loss with class weights")
    print("-"*60)
    alpha = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])  # 희귀 클래스에 높은 가중치
    focal_loss_weighted = FocalLoss(alpha=alpha, gamma=2.0)
    loss2 = focal_loss_weighted(logits, targets)
    print(f"✅ Loss with weights: {loss2.item():.4f}")
    
    # Test 3: Weighted Focal Loss (자동 가중치)
    print("\nTest 3: Weighted Focal Loss (auto weights)")
    print("-"*60)
    class_counts = np.array([50, 25, 15, 8, 2])  # 샘플 개수
    weighted_focal = WeightedFocalLoss(class_counts=class_counts, gamma=2.0, beta=0.99)
    loss3 = weighted_focal(logits, targets)
    print(f"✅ Loss: {loss3.item():.4f}")
    
    # Test 4: Adaptive Focal Loss
    print("\nTest 4: Adaptive Focal Loss")
    print("-"*60)
    adaptive_focal = AdaptiveFocalLoss(alpha=alpha, gamma_start=3.0, gamma_end=1.0, total_epochs=10)
    print(f"Initial gamma: {adaptive_focal.get_current_gamma():.2f}")
    
    for epoch in range(5):
        adaptive_focal.update_gamma(epoch)
        loss4 = adaptive_focal(logits, targets)
        print(f"  Epoch {epoch}: gamma={adaptive_focal.get_current_gamma():.2f}, loss={loss4.item():.4f}")
    
    # Comparison with CE Loss
    print("\nTest 5: Comparison with Cross Entropy")
    print("-"*60)
    ce_loss = F.cross_entropy(logits, targets)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Focal Loss:         {loss1.item():.4f}")
    print(f"Difference:         {abs(loss1.item() - ce_loss.item()):.4f}")
    
    print("\n" + "="*60)
    print("✅ All Focal Loss tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_focal_loss()
