"""
Week 8: Attention Mechanism Unit Tests

Multi-head Self-Attention 모듈 단위 테스트
- ScaledDotProductAttention
- MultiHeadAttention  
- PositionalEncoding
- EncoderLayer
- PitchTransformerModel

Author: AI Pitch Sequencing Team
Date: 2025-01-XX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    EncoderLayer,
    PitchTransformerModel
)


def test_scaled_dot_product_attention():
    """Test 1: Scaled Dot-Product Attention"""
    print("\n" + "=" * 70)
    print("Test 1: Scaled Dot-Product Attention")
    print("=" * 70)
    
    batch_size = 4
    n_heads = 2
    seq_len = 5
    d_k = 8
    
    # Create attention module
    attention = ScaledDotProductAttention(dropout=0.1)
    attention.eval()  # Set to eval mode to disable dropout
    
    # Create Q, K, V
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    print(f"Input shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    
    # Forward pass
    output, weights = attention(Q, K, V)
    
    print(f"\nOutput shapes:")
    print(f"  output: {output.shape}")
    print(f"  weights: {weights.shape}")
    
    # Validate shapes
    assert output.shape == (batch_size, n_heads, seq_len, d_k), "Output shape mismatch"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), "Weights shape mismatch"
    
    # Validate attention weights sum to 1
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5), "Attention weights don't sum to 1"
    
    print("\n✅ Test 1 Passed: Scaled Dot-Product Attention")


def test_multi_head_attention():
    """Test 2: Multi-head Attention"""
    print("\n" + "=" * 70)
    print("Test 2: Multi-head Attention")
    print("=" * 70)
    
    batch_size = 4
    seq_len = 10
    d_model = 128
    n_heads = 4
    
    # Create module
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    mha.eval()  # Set to eval mode to disable dropout
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass (self-attention: Q=K=V=x)
    output, weights = mha(x, x, x)
    
    print(f"\nOutput shapes:")
    print(f"  output: {output.shape}")
    print(f"  weights: {weights.shape}")
    
    # Validate shapes
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), "Weights shape mismatch"
    
    # Validate parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Expected: W_q, W_k, W_v, W_o = 4 × (d_model × d_model + d_model)
    expected_params = 4 * (d_model * d_model + d_model)
    assert total_params == expected_params, f"Parameter count mismatch: {total_params} vs {expected_params}"
    
    print("\n✅ Test 2 Passed: Multi-head Attention")


def test_positional_encoding():
    """Test 3: Positional Encoding"""
    print("\n" + "=" * 70)
    print("Test 3: Positional Encoding")
    print("=" * 70)
    
    batch_size = 4
    seq_len = 20
    d_model = 128
    
    # Create module
    pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)  # No dropout for testing
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = pe(x)
    
    print(f"Output shape: {output.shape}")
    
    # Validate shape
    assert output.shape == x.shape, "Shape mismatch"
    
    # Validate that output != input (positional encoding added)
    # Note: Due to dropout, we need to set eval mode
    pe.eval()
    output_eval = pe(x)
    assert not torch.equal(x, output_eval), "Positional encoding not added"
    
    # Visualize first position encoding
    print(f"\nFirst 10 positions of PE (dim 0-5):")
    with torch.no_grad():
        pe_values = pe.pe[0, :10, :6].numpy()
        for i, pos_enc in enumerate(pe_values):
            print(f"  pos {i}: {pos_enc}")
    
    print("\n✅ Test 3 Passed: Positional Encoding")


def test_encoder_layer():
    """Test 4: Transformer Encoder Layer"""
    print("\n" + "=" * 70)
    print("Test 4: Transformer Encoder Layer")
    print("=" * 70)
    
    batch_size = 4
    seq_len = 10
    d_model = 128
    n_heads = 4
    d_ff = 512
    
    # Create module
    encoder = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1)
    encoder.eval()  # Set to eval mode to disable dropout
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output, weights = encoder(x)
    
    print(f"\nOutput shapes:")
    print(f"  output: {output.shape}")
    print(f"  weights: {weights.shape}")
    
    # Validate shapes
    assert output.shape == x.shape, "Output shape mismatch"
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len), "Weights shape mismatch"
    
    # Validate parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ Test 4 Passed: Encoder Layer")


def test_pitch_transformer_model():
    """Test 5: Full PitchTransformerModel"""
    print("\n" + "=" * 70)
    print("Test 5: PitchTransformerModel (Full Model)")
    print("=" * 70)
    
    # Hyperparameters
    INPUT_SIZE = 39
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 512
    NUM_CLASSES = 4
    BATCH_SIZE = 8
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
    
    print(f"Model created with:")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  d_model: {D_MODEL}")
    print(f"  n_heads: {N_HEADS}")
    print(f"  n_layers: {N_LAYERS}")
    
    # Test with sequence input
    print(f"\n--- Test with sequence input ---")
    x_seq = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
    print(f"Input: {x_seq.shape}")
    
    logits_seq, attn_seq = model(x_seq)
    
    print(f"Output: {logits_seq.shape}")
    print(f"Attention layers: {len(attn_seq)}")
    
    assert logits_seq.shape == (BATCH_SIZE, NUM_CLASSES), "Logits shape mismatch"
    assert len(attn_seq) == N_LAYERS, "Number of attention layers mismatch"
    
    for layer_name, weights in attn_seq.items():
        print(f"  {layer_name}: {weights.shape}")
        assert weights.shape == (BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN), f"{layer_name} weights shape mismatch"
    
    # Test with single pitch input (no sequence)
    print(f"\n--- Test with single pitch input ---")
    x_single = torch.randn(BATCH_SIZE, INPUT_SIZE)
    print(f"Input: {x_single.shape}")
    
    logits_single, attn_single = model(x_single)
    
    print(f"Output: {logits_single.shape}")
    
    assert logits_single.shape == (BATCH_SIZE, NUM_CLASSES), "Single pitch logits shape mismatch"
    
    # Test predictions
    print(f"\n--- Test predictions ---")
    probs = F.softmax(logits_seq, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    pitch_types = ['FF', 'SL', 'CH', 'CU']
    
    print(f"\nSample predictions (first 3):")
    for i in range(min(3, BATCH_SIZE)):
        pred_pitch = pitch_types[preds[i]]
        prob_str = ' '.join([f'{pitch_types[j]}={probs[i,j]:.3f}' for j in range(NUM_CLASSES)])
        print(f"  {i+1}. Pred: {pred_pitch} | Probs: {prob_str}")
    
    # Test gradient flow
    print(f"\n--- Test gradient flow ---")
    loss = F.cross_entropy(logits_seq, torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)))
    loss.backward()
    
    # Check if gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    assert has_grad == total_params, "Some parameters don't have gradients"
    
    # Model parameter count
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_param_count:,}")
    
    print("\n✅ Test 5 Passed: PitchTransformerModel")


def test_attention_visualization():
    """Test 6: Attention Weights Visualization"""
    print("\n" + "=" * 70)
    print("Test 6: Attention Weights Visualization")
    print("=" * 70)
    
    # Small model for visualization
    model = PitchTransformerModel(
        input_size=39,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=128,
        num_classes=4,
        dropout=0.0  # No dropout for deterministic output
    )
    
    # Single batch, short sequence
    x = torch.randn(1, 5, 39)
    
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(x)
    
    # Extract layer 0 attention weights
    layer0_attn = attn_weights['layer_0']  # (1, n_heads, seq_len, seq_len)
    
    print(f"Attention weights shape: {layer0_attn.shape}")
    print(f"\nAttention matrix (head 0):")
    
    # Show attention matrix for first head
    attn_matrix = layer0_attn[0, 0].numpy()  # (seq_len, seq_len)
    
    print("     ", end="")
    for j in range(attn_matrix.shape[1]):
        print(f"pos{j:1d}   ", end="")
    print()
    
    for i in range(attn_matrix.shape[0]):
        print(f"pos{i}: ", end="")
        for j in range(attn_matrix.shape[1]):
            print(f"{attn_matrix[i,j]:.3f} ", end="")
        print()
    
    # Validate attention weights sum to 1 for each query
    attn_sum = attn_matrix.sum(axis=1)
    print(f"\nRow sums (should be 1.0): {attn_sum}")
    assert np.allclose(attn_sum, 1.0, atol=1e-5), "Attention weights don't sum to 1"
    
    print("\n✅ Test 6 Passed: Attention Visualization")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Week 8: Attention Mechanism Unit Tests")
    print("=" * 70)
    
    try:
        test_scaled_dot_product_attention()
        test_multi_head_attention()
        test_positional_encoding()
        test_encoder_layer()
        test_pitch_transformer_model()
        test_attention_visualization()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
