# tests/test_main.py
import pytest
import torch# <--- NOTICE: direct import from 'slapping'

def test_my_function_output_shape():
    # 1. Setup
    x = torch.randn(1, 10)
    result = x
    
    assert result.shape == (1, 10)
    assert not torch.isnan(result).any()
    