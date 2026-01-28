import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn

# Adapted from https://github.com/automl/TransformersCanDoBayesianInference
class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)
    



class PriorTransformerModel(nn.Module):
    def __init__(self, encoder, n_out, ninp, nhead, nhid, nlayers, dropout, num_test_parameters, num_features, y_encoder=None, pos_encoder=None, input_normalization=False):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.num_test_parameters = num_test_parameters
        self.num_features = num_features
        self.x_test = nn.Parameter(data=torch.randn(num_test_parameters, ninp))
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder  
        self.decoder = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.decoder_params = nn.Sequential(nn.Linear(ninp, nhid), nn.ReLU(), nn.Linear(nhid, n_out))

        self.init_weights()

    @staticmethod
    def generate_square_subsequent_mask(sz):
        '''
        Docstring for generate_square_subsequent_mask, but not permutation equivariant
        
        :param sz: Size of the matrix
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        '''
        Docstring for generate_D_q_matrix - attention matrix, but permutation equivariant
        
        :param sz: Size of the matrix
        :param query_size: Size of the test set 
        '''
        train_size = sz-query_size
        mask = torch.zeros(sz,sz) == 0
        mask[:,train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def forward(self, src, src_mask=None, context_pos=None):
        only_x = not isinstance(src, tuple) # We are passing just X, without training context    

        if isinstance(src, tuple):
            x_src, y_src = src
            x_src = self.encoder(x_src)
            y_src = self.y_encoder(y_src.unsqueeze(-1))
            if context_pos is None:
                src = torch.stack([x_src, y_src], 1).view(-1, *x_src.shape[1:])
                ## (2, XDIM) - One dim for X, one for Y. X1, Y1, X2, Y2 - X1 can't see Y1, but X2 can see X1, Y1
            else:
                # X and Y are embedded together as context 
                train_x = x_src[:context_pos] + y_src[:context_pos] 
                src = torch.cat([train_x, x_src[context_pos:]], 0)
        else:
            src = self.encoder(src)
            
        is_unbatched = src.dim() == 2
        if is_unbatched:
            src = src.unsqueeze(1) # Convert (Seq, Dim) -> (Seq, 1, Dim)
            
        batch_size = src.shape[1]
        x_test_batch = self.x_test.unsqueeze(1).repeat(1, batch_size, 1)
        src = torch.cat([src, x_test_batch], dim=0)
        total_len = len(src)
        
        if src_mask is None:
            if context_pos is None:
                # AR Case: standard causal mask for the whole sequence
                src_mask = self.generate_square_subsequent_mask(total_len).to(src.device)
            else:
                src_mask = self.generate_D_q_matrix(total_len, total_len - context_pos).to(src.device)
        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output_xy = self.decoder(output[:-len(x_test_batch)])
        output_params = self.decoder_params(output[-len(x_test_batch):])
        output = torch.cat([output_xy, output_params])

        if only_x:
            return output 
        elif context_pos is None:
            return output[0::2] # We only look at the X positions' predictions 
        else:
            return output[context_pos:]