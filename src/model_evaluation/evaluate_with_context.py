
import torch

def evaluate_parameter_distributions_on_model(train_X, train_Y, model, device, num_training_points):
    model = model.to(device)
    train_x = train_X
    train_y = train_Y
    with torch.no_grad():
        logits = model((train_x, train_y), context_pos=num_training_points)
        outputs = torch.exp(torch.log_softmax(logits, -1))
    
    return outputs