from functools import partial
from transformers import LlamaForSequenceClassification
import torch
from .ssm_lora import LinearWithLoRA

def prepare_model(config, time_axis):
    """Prepare model with SSM-LoRA"""
    model = LlamaForSequenceClassification.from_pretrained(
        config.model_path, num_labels=config.num_labels
    )
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
    assign_lora = partial(
        LinearWithLoRA, 
        rank=config.lora_r, 
        alpha=config.lora_alpha, 
        time_axis=time_axis,
        device=device
    )

    # Apply LoRA to different components
    for i, layer in enumerate(model.model.layers):
        if i % 2 == 0 and config.lora_query:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj, layer_key=f"query_{i}")
        elif config.lora_value:
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj, layer_key=f"value_{i}")
        
        if config.lora_projection:
            layer.self_attn.o_proj = assign_lora(layer.self_attn.o_proj, layer_key="od")
        
        if config.lora_mlp:
            layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj, layer_key="mlp")
            layer.mlp.up_proj = assign_lora(layer.mlp.up_proj, layer_key="mu")
            layer.mlp.down_proj = assign_lora(layer.mlp.down_proj, layer_key="md")
    
    if config.lora_head:
        model.score = assign_lora(model.score, layer_key="score")

    return model

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)