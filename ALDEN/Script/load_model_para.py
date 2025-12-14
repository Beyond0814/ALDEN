"""
Utility functions for calculating and displaying model parameters.
"""
import logging

def Load_ModelPara(model):  
    """
    Calculate and log the total number of parameters in a model.
    
    Args:
        model: PyTorch model
    """
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    nb_para, nb_s = para_cal(nb_params)
    logging.info(f'Number of total parameters: {nb_para:.2f}{nb_s}')
    learn_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    learn_para, learn_s = para_cal(learn_params)
    logging.info(f'Number of learnable parameters: {learn_para:.2f}{learn_s}')

def get_model_parameters_info(model):
    """
    Print detailed parameter information for each layer in the model.
    
    Args:
        model: PyTorch model
    """
    logging.info(f"{'Layer':<30} {'Total Params':<15} {'Trainable Params':<20} {'Frozen Params'}")
    logging.info("="*80)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only consider leaf nodes
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            logging.info(f"{name:<30} {total_params:<15} {trainable_params:<20} {frozen_params}")


def para_cal(nb_params):
    """
    Convert parameter count to human-readable format (K, M, G, T).
    
    Args:
        nb_params: Number of parameters
        
    Returns:
        Tuple of (scaled_value, scale_unit)
    """
    scale = ('', 'K', 'M', 'G', 'T')
    scaled_param = nb_params
    for s in scale:
        if scaled_param < 1000:
            break
        scaled_param /= 1024
    return scaled_param, s
