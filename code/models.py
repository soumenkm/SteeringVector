import os, torch, json, sys
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.manual_seed(42)
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig
from typing import List, Tuple, Union
import pandas as pd
from utils import models_map
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

class LoRALayer(torch.nn.Module):
    def __init__(self, rank: int, alpha: float, d_in: int, d_out: int, mask_A: Union[torch.Tensor, None], mask_B: Union[torch.Tensor, None]):  
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
        self.mask_A = mask_A
        self.mask_B = mask_B
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        masked_A = self.A.to(self.mask_A.device) * self.mask_A
        masked_B = self.B.to(self.mask_A.device) * self.mask_B
        delta_W = torch.matmul(masked_A, masked_B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        return z
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float, mask_A: Union[torch.Tensor, None], mask_B: Union[torch.Tensor, None]):
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        self.lora = LoRALayer(rank=self.rank, alpha=self.alpha, d_in=self.d_in, d_out=self.d_out, mask_A=mask_A, mask_B=mask_B)

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
        return z

class ModelForCLS(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, num_class: int, quant_config: Union[BitsAndBytesConfig, None]):
        super(ModelForCLS, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base = AutoModel.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.d = self.base.config.hidden_size
        self.c = num_class
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Multi-layer classification head with ReLU activation
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d, self.c)       
        ).to(self.device)
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor]) -> torch.tensor:
        z = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (b, T, d)
        y = z[:, -1, :].to(self.device) # (b, d) (last position embedding)
        out = self.head(y) # (b, c)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = self.loss_fn(out, labels) 
            acc = (out.argmax(dim=-1) == labels).to(torch.float32).mean().item()
        else:
            loss = None
            acc = None
        return {"logits": out, "loss": loss, "acc": acc}

class ModelForCLSWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, tokenizer: AutoTokenizer, model_name: str, sparse_alpha: Union[float, None], num_class: int, lora_rank: int, lora_alpha: float, quant_config: Union[None, BitsAndBytesConfig], frozen_neurons: Union[None, torch.tensor]):
        super(ModelForCLSWithLoRA, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = tokenizer
        self.num_class = num_class
        if sparse_alpha is None:
            self.model = ModelForCLS(device=self.device, model_name=self.model_name, num_class=self.num_class, quant_config=quant_config)
        else:
            self.model = SparseModelForCLS(device=self.device, model_name=self.model_name, num_class=self.num_class, quant_config=quant_config, alpha=sparse_alpha)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.base.config.pad_token_id = self.tokenizer.pad_token_id
        
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layer_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        elif "bloomz" in m:
            # layer_names = ['query_key_value', 'dense']
            layer_names = ['query_key_value', 'dense', 'dense_4h_to_h']
        else:
            raise NotImplementedError("Invalid model name!")
        self.apply_lora(rank=self.rank, alpha=self.alpha, frozen_neurons=frozen_neurons, layer_names=layer_names)
    
    def apply_lora(self, rank: int, alpha: float, frozen_neurons: Union[None, torch.tensor], layer_names: List[str]) -> None:
        """frozen_neuron_ids = [(i, j) | j: neuron index for a layer i]
        """
        # if frozen_neurons is not None:
        #     frozen_neurons_dict = {}
        #     for (layer_idx, neuron_idx) in frozen_neurons:
        #         layer_idx = layer_idx.item()
        #         if layer_idx not in frozen_neurons_dict:
        #             frozen_neurons_dict[layer_idx] = []
        #         frozen_neurons_dict[layer_idx].append(neuron_idx.item())

        #     for layer_idx, layer in enumerate(self.get_layers()):
        #         target_linear_name, target_module = self.get_target_linear_module(layer_idx=layer_idx)
        #         if layer_idx in frozen_neurons_dict:
        #             frozen_neuron_ids = frozen_neurons_dict[layer_idx]
        #             up_proj_linear = getattr(target_module, target_linear_name)
        #             mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=self.device, d_in=up_proj_linear.in_features, d_out=up_proj_linear.out_features, rank=rank, frozen_neuron_ids=frozen_neuron_ids)
        #             up_proj_lora = LinearWithLoRA(up_proj_linear, rank, alpha, mask_A, mask_B)
        #             setattr(target_module, target_linear_name, up_proj_lora)
                    
        ModelForCLSWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.base, rank=rank, alpha=alpha, layer_names=layer_names)            

    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float, layer_names: List[str]):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                if any(proj in name for proj in layer_names):
                    mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=device, d_in=module.in_features, d_out=module.out_features, rank=rank, frozen_neuron_ids=None)
                    linear_lora = LinearWithLoRA(module, rank, alpha, mask_A, mask_B)
                    setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForCLSWithLoRA.replace_linear_with_lora(device, module, rank, alpha, layer_names)
    
    @staticmethod
    def create_lora_mask(device: torch.device, d_in: int, d_out: int, rank: int, frozen_neuron_ids: Union[None, torch.tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """frozen_neuron_ids = [j | j: neuron index for a layer i]
        """
        mask_A = torch.ones(d_in, rank).to(device)
        mask_B = torch.ones(rank, d_out).to(device)
        if frozen_neuron_ids is not None:
            for neuron_id in frozen_neuron_ids:
                mask_B[:, neuron_id] = 0  # Zero out the entire column for the frozen neurons
        return mask_A, mask_B
     
    def calc_num_lora_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
    
    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.base.layers
        elif "bloomz" in m:
            layers = self.model.base.h
        else:
            raise NotImplementedError("Invalid model name!")
        return layers
    
    def get_target_act_module(self, layer_idx: int) -> torch.nn.Module:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp.act_fn
        elif "bloomz" in m:
            target_module = mlp.gelu_impl
        else:
            raise NotImplementedError("Invalid model name!")
        return target_module

    def get_target_linear_module(self, layer_idx: int) -> Tuple[str, torch.nn.Linear]:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp        
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp
            name = "gate_proj"
        elif "bloom" in m:
            target_module = mlp
            name = "dense_h_to_4h"
        else:
            raise NotImplementedError("Invalid model name!")
        return name, target_module
    
    def create_hook_function(self, intervene_config: dict):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            indices = intervene_config["indices"] # List[n x Tuple[layer_index, neuron_index]] (n neurons to intervene)
            value = intervene_config["value"] # List[n] 
            for i, (layer_idx, neuron_idx) in enumerate(indices):
                if module == self.get_target_act_module(layer_idx=layer_idx):
                    outputs.index_fill_(dim=-1, index=torch.tensor([neuron_idx]).to(self.device), value=value[i])
        return hook_function

    def register_hook(self, intervene_config: Union[dict, None]):
        self.hooks_list = [] # List[L]
        for layer_idx in range(len(self.get_layers())):
            hook_function = self.create_hook_function(intervene_config=intervene_config)
            h = self.get_target_act_module(layer_idx=layer_idx).register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, intervene_config: Union[dict, None] = None, labels: Union[torch.tensor, None] = None) -> torch.tensor:
        """intervene_config = {
            "indices": [(i, j) | i: layer index, j: neuron index]
            "value": [i | for each index i in indices])
        }"""
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        if intervene_config is not None:
            self.register_hook(intervene_config=intervene_config)
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, c)
            self.remove_hook()
        else:
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # (b, c)
        return prediction_output

class ModelForCLM(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, quant_config: Union[BitsAndBytesConfig, None]):
        super(ModelForCLM, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.base = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.d = self.base.config.hidden_size
        # self.base.resize_token_embeddings(len(self.tokenizer))
        self.base.config.pad_token_id = self.tokenizer.pad_token_id
    
    def compute_metrics_manually(self, z, labels, acc_mask):
        # z.shape = (b, T, V), labels.shape = (b, T), acc_mask.shape = (b, T)
        pred_prob = torch.nn.functional.softmax(z, dim=-1) # (b, T, V) 
        i, j = torch.where(labels != -100)
        filter_pred_prob = pred_prob[i, j] # (N, V)
        filter_true_class = labels[labels != -100] # (N,)
        filter_true_prob = torch.nn.functional.one_hot(filter_true_class, num_classes=filter_pred_prob.shape[-1]) # (N, V)
        cross_entropy = (-filter_true_prob * torch.log(filter_pred_prob)).sum(dim=1).mean()
        
        i, j = torch.where(acc_mask != 0)
        filter_pred_prob = pred_prob[i, j] # (N, V)
        filter_true_class = labels[acc_mask != 0] # (N,)
        filter_true_prob = torch.nn.functional.one_hot(filter_true_class, num_classes=filter_pred_prob.shape[-1]) # (N, V)
        cross_entropy_ans = (-filter_true_prob * torch.log(filter_pred_prob)).sum(dim=1).mean()
        acc = (filter_pred_prob.argmax(dim=-1) == filter_true_class).to(torch.float32).mean()
        return {"loss": cross_entropy.item(), "loss_ans": cross_entropy_ans.item(), "acc": acc.item()}
     
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: Union[None, torch.tensor], acc_mask: Union[None, torch.tensor]) -> torch.tensor:
        z = self.base(input_ids=input_ids, attention_mask=attention_mask).logits # (b, T, d)
        if labels is not None:
            labels = labels.to(self.device)  
            loss = torch.nn.functional.cross_entropy(z.flatten(0, 1), labels.flatten(), ignore_index=-100) 
            acc = self.compute_metrics_manually(z, labels, acc_mask)["acc"]
        else:
            loss = None
            acc = None
        return {"logits": z, "loss": loss, "acc": acc}

class ModelForCLMWithLoRA(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, lora_rank: int, lora_alpha: float, quant_config: Union[None, BitsAndBytesConfig], frozen_neurons):
        super(ModelForCLMWithLoRA, self).__init__()
        self.model_name = model_name
        self.device = device
        self.frozen_neurons = frozen_neurons
        self.model = ModelForCLM(device=self.device, model_name=self.model_name, quant_config=quant_config)
        
        for param in self.model.base.parameters():
            param.requires_grad = False
        
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.apply_lora(rank=self.rank, alpha=self.alpha, frozen_neurons=self.frozen_neurons)
    
    def apply_lora(self, frozen_neurons, rank: int, alpha: float) -> None:
        if frozen_neurons is not None:
            frozen_neurons_dict = {}
            for (layer_idx, neuron_idx) in frozen_neurons:
                layer_idx = layer_idx.item()
                if layer_idx not in frozen_neurons_dict:
                    frozen_neurons_dict[layer_idx] = []
                frozen_neurons_dict[layer_idx].append(neuron_idx.item())

            for layer_idx, layer in enumerate(self.get_layers()):
                target_linear_name, target_module = self.get_target_linear_module(layer_idx=layer_idx)
                if layer_idx in frozen_neurons_dict:
                    frozen_neuron_ids = frozen_neurons_dict[layer_idx]
                    up_proj_linear = getattr(target_module, target_linear_name)
                    mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=self.device, d_in=up_proj_linear.in_features, d_out=up_proj_linear.out_features, rank=rank, frozen_neuron_ids=frozen_neuron_ids)
                    up_proj_lora = LinearWithLoRA(up_proj_linear, rank, alpha, mask_A, mask_B)
                    setattr(target_module, target_linear_name, up_proj_lora)
                
        ModelForCLMWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.base, rank=rank, alpha=alpha, layer_names=['q_proj', 'k_proj', 'v_proj'])            
        # ModelForCLMWithLoRA.replace_linear_with_lora(device=self.device, model=self.model.base, rank=rank, alpha=alpha, layer_names=["lm_head"])            

    @staticmethod
    def replace_linear_with_lora(device: torch.device, model: torch.nn.Module, rank: int, alpha: float, layer_names: List[str]):
        """layer_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']"""
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                if any(proj in name for proj in layer_names):
                    mask_A, mask_B = ModelForCLSWithLoRA.create_lora_mask(device=device, d_in=module.in_features, d_out=module.out_features, rank=rank, frozen_neuron_ids=None)
                    linear_lora = LinearWithLoRA(module, rank, alpha, mask_A, mask_B)
                    setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ModelForCLMWithLoRA.replace_linear_with_lora(device, module, rank, alpha, layer_names)
    
    @staticmethod
    def create_lora_mask(device: torch.device, d_in: int, d_out: int, rank: int, frozen_neuron_ids: Union[None, torch.tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """frozen_neuron_ids = [j | j: neuron index for a layer i]
        """
        mask_A = torch.ones(d_in, rank).to(device)
        mask_B = torch.ones(rank, d_out).to(device)
        if frozen_neuron_ids is not None:
            for neuron_id in frozen_neuron_ids:
                mask_B[:, neuron_id] = 0  # Zero out the entire column for the frozen neurons
        return mask_A, mask_B
     
    def calc_num_params(self) -> None:
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
    
    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.base.model.layers
        elif "bloom" in m:
            layers = self.model.base.model.transformer.h
        else:
            raise NotImplementedError("Invalid model name!")
        return layers
    
    def get_target_act_module(self, layer_idx: int) -> torch.nn.Module:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp.act_fn
        elif "bloom" in m:
            target_module = mlp.gelu_impl
        else:
            raise NotImplementedError("Invalid model name!")
        return target_module

    def get_target_linear_module(self, layer_idx: int) -> Tuple[str, torch.nn.Linear]:
        m = self.model_name.lower()
        mlp = self.get_layers()[layer_idx].mlp        
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            target_module = mlp
            name = "gate_proj"
        elif "bloom" in m:
            target_module = mlp
            name = "dense_h_to_4h"
        else:
            raise NotImplementedError("Invalid model name!")
        return name, target_module
    
    def create_hook_function(self, intervene_config: dict):
        def hook_function(module: torch.nn.Module, inputs: torch.Tensor, outputs: torch.Tensor):
            indices = intervene_config["indices"] # List[n x Tuple[layer_index, neuron_index]] (n neurons to intervene)
            value = intervene_config["value"] # List[n] 
            for i, (layer_idx, neuron_idx) in enumerate(indices):
                if module == self.get_target_act_module(layer_idx=layer_idx):
                    outputs.index_fill_(dim=-1, index=torch.tensor([neuron_idx]).to(self.device), value=value[i])
        return hook_function

    def register_hook(self, intervene_config: Union[dict, None]):
        self.hooks_list = [] # List[L]
        for layer_idx in range(len(self.get_layers())):
            hook_function = self.create_hook_function(intervene_config=intervene_config)
            h = self.get_target_act_module(layer_idx=layer_idx).register_forward_hook(hook_function)
            self.hooks_list.append(h)
    
    def remove_hook(self):
        for h in self.hooks_list:
            h.remove()
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, intervene_config: Union[dict, None] = None, labels: Union[torch.tensor, None] = None, acc_mask: Union[None, torch.tensor] = None) -> torch.tensor:
        """intervene_config = {
            "indices": [(i, j) | i: layer index, j: neuron index]
            "value": [i | for each index i in indices])
        }"""
        assert list(input_ids.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        if intervene_config is not None:
            self.register_hook(intervene_config=intervene_config)
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, acc_mask=acc_mask) # (b, c)
            self.remove_hook()
        else:
            prediction_output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, acc_mask=acc_mask) # (b, c)
        return prediction_output

class ContrastiveSteeringModule(torch.nn.Module):
    def __init__(self, device: torch.device, hidden_dim: int, head_dim: int):
        super(ContrastiveSteeringModule, self).__init__()
        self.device = device
        self.d = hidden_dim
        self.k = head_dim
        self.mlp_W = torch.nn.Linear(in_features=self.d, out_features=1).to(self.device)
        self.mlp_K = torch.nn.Linear(in_features=self.d, out_features=self.k).to(self.device)
        self.tau = 1.0
    
    def infoNCELoss(self, x: torch.tensor) -> torch.tensor:
        """x.shape = (b, k)"""
        sim = [torch.exp(torch.nn.functional.cosine_similarity(x[0], x[i], dim=0)/self.tau) for i in range(1, x.shape[0])]
        sim_tensor = torch.stack(sim) # (b-1,)
        loss = -torch.log(sim_tensor[0]/sim_tensor.sum(dim=0)) # scalar
        return loss
        
    def forward(self, x: torch.tensor, padding_mask: torch.tensor) -> dict:
        """z.shape = (b, T, d), padding_mask.shape = (b, T) [0 means pad and 1 means non pad]"""
        
        x = x.to(self.device)
        padding_mask = padding_mask.to(self.device)
        w = self.mlp_W(x).squeeze() # (b, T, 1) -> (b, T)
        w_mod = torch.where(padding_mask == 1, w, -float('inf')) # (b, T)
        a = torch.nn.functional.softmax(w_mod, dim=-1) # (b, T)
        z = torch.bmm(a.unsqueeze(1), x).squeeze(1) # (b, d)
        y = self.mlp_K(z) # (b, k)
        
        loss = self.infoNCELoss(x=y) 
        acc = None
        return {"loss": loss, "acc": acc}

class LayerWithAdapter(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, adapter: torch.nn.Module):
        super(LayerWithAdapter, self).__init__()
        self.layer = layer
        self.adapter = adapter
        self.adapter_output = None

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs) # (1, b, T, d)
        adapter_out = self.adapter(output[0], padding_mask=kwargs["padding_mask"]) # dict which has loss
        self.adapter_output = adapter_out
        return output

class ModelForContrastiveSteering(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str, head_dim: int, steering_layer_list: List[int], quant_config: Union[BitsAndBytesConfig, None]):
        super(ModelForContrastiveSteering, self).__init__()
        self.device = device
        self.model_name = model_name
        self.model_name_srt = model_name.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quant_config, device_map="auto")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.d = self.model.config.hidden_size
        self.h = head_dim
        self.steering_layer_list = steering_layer_list
        for i in self.steering_layer_list:
            csm = ContrastiveSteeringModule(device=self.device, hidden_dim=self.d, head_dim=self.h)
            self.modify_layer(adapter=csm, layer_index=i)

    def get_layers(self) -> torch.nn.ModuleList:
        m = self.model_name.lower()
        if ("llama" in m) or ("mistral" in m) or ("sarvam" in m) or ("aya-23" in m):
            layers = self.model.model.layers
        elif "bloomz" in m:
            layers = self.model.transformer.h
        else:
            raise NotImplementedError("Invalid model name!")
        return layers
    
    def modify_layer(self, adapter: torch.tensor, layer_index: int) -> None:
        layers = self.get_layers()
        original_layer = layers[layer_index]
        layers[layer_index] = LayerWithAdapter(original_layer, adapter)
    
    def compute_metric(self, inputs: torch.tensor, target: torch.tensor) -> dict:
        loss1 = torch.nn.functional.cross_entropy(inputs.flatten(0, 1), target.flatten(), ignore_index=-100) 
        layers = self.get_layers()
        loss2 = torch.stack([layers[i].adapter_output["loss"] for i in self.steering_layer_list]).sum()
        loss = loss1 + loss2
        acc = None
        return {"loss": loss, "acc": acc}  
    
    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, labels: torch.tensor, padding_mask: torch.tensor) -> torch.tensor:
        z = self.model(input_ids=input_ids, attention_mask=attention_mask, padding_mask=padding_mask).logits # (b, T, d)
        labels = labels.to(self.device)  
        metrics = self.compute_metric(inputs=z, target=labels)
        return {"logits": z, **metrics}

def main_clm(model_name: str, device: torch.device) -> None:
    quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
    )
    # quant_config = None
    model = ModelForContrastiveSteering(device=device, model_name=model_name, quant_config=quant_config, head_dim=64, steering_layer_list=[-1]).to(device)
    
    input_ids = torch.randint(low=0, high=1000, size=(16, 513), dtype=torch.int64).to(device)
    attention_mask = torch.randint(low=0, high=2, size=(16, 513), dtype=torch.int64).to(device)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    padding_mask = attention_mask.clone()
    
    # model = ContrastiveSteeringModule(device=device, head_dim=64, hidden_dim=4096).to(device)
    
    print(model)
    model.train()
    with torch.autocast("cuda"):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, padding_mask=padding_mask) # (b, V)
        # out = model(x=torch.rand(size=(16, 512, 4096)), padding_mask=torch.cat([torch.ones(size=(16, 500)), torch.zeros(size=(16, 12))], dim=-1))
    print(out["logits"].max(), out["logits"].min(), out["loss"], out["acc"])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main_clm(models_map["llama3"], device=device)