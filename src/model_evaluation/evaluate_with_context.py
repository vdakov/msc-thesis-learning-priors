
import torch

def evaluate_parameter_distributions_on_model(train_X, train_Y, model, device, num_training_points):
    model = model.to(device)
    with torch.no_grad():
        logits = model((train_X, train_Y), context_pos=num_training_points)
        outputs = torch.exp(torch.log_softmax(logits, -1))
    
    return outputs