from criterion.bar_distribution import BarDistribution, FullSupportBarDistribution 
import pytest 
import torch 
from torch import nn 


@pytest.fixture
def simple_borders():
    return torch.tensor([0.0, 1.0, 2.0, 3.0])

@pytest.fixture
def unsorted_borders():
    return torch.tensor([0.0, -1.0, 1.0, 3.0])

@pytest.fixture 
def different_width_borders():
    return torch.tensor([0.0, 5.0, 2.0, 3.0])

@pytest.fixture
def bar_dist(simple_borders):
    return BarDistribution(simple_borders)

@pytest.fixture
def full_support_dist(simple_borders):
    return FullSupportBarDistribution(simple_borders)

@pytest.fixture 
def logits():
    return torch.tensor([[-10.0, -10.0, 10.0]])

class TestBarDistribution:
    def test_init(self, simple_borders):
        bd = BarDistribution(simple_borders)
        assert bd.num_bars == 3 
        assert torch.equal(bd.bucket_widths, torch.tensor([1, 1, 1]))
    
    def test_init_fails_on_mutlidim_borders(self, simple_borders):
        multidim_bord = torch.unsqueeze(simple_borders, 0)
        
        with pytest.raises(AssertionError, match="Borders should be one-dimensional"):
            BarDistribution(multidim_bord)
        
    def test_init_fails_on_unsorted_borders(self, unsorted_borders):
        with pytest.raises(
            AssertionError, match="Please provide sorted borders",):
            BarDistribution(unsorted_borders)
    
    def test_init_fails_on_diff_witdh_borders(self, different_width_borders):
        with pytest.raises(AssertionError):
            BarDistribution(different_width_borders)
            
    def test_map_to_bucket_idx(self, bar_dist):
        assert bar_dist.map_to_bucket_idx(1.5) == 1
        assert torch.equal(bar_dist.map_to_bucket_idx(torch.tensor([0.5, 1.5, 2.5])),torch.tensor([0, 1, 2]))
        
    def test_mean(self, bar_dist, logits): 
        pred_mean = bar_dist.mean(logits)
        assert torch.isclose(pred_mean, torch.tensor([2.5]), atol=1e-1)
        uniform_logits = torch.zeros((1, 3))
        uniform_mean = bar_dist.mean(uniform_logits)
        assert torch.isclose(uniform_mean, torch.tensor([1.5]), atol=1e-1)
    
    def test_forward_shape_and_value(self, bar_dist):
        # Setup: 1 batch, 1 target sample
        # Logits equal -> Uniform distribution
        logits = torch.zeros(1, 1, 3)
        y = torch.tensor([[0.5]])  # Belongs to bucket 0

        # Loss calculation:
        # Prob of bucket 0 = softmax(0,0,0) = 1/3
        # Width = 1
        # Density = (1/3) / 1 = 1/3
        # NLL = -log(1/3) = log(3) ≈ 1.0986

        loss = bar_dist(logits, y)

        assert loss.shape == (1, 1)
        expected_loss = torch.log(torch.tensor(3.0))
        assert torch.allclose(loss, expected_loss, atol=1e-4)
        
    