import torch.nn as nn
import torch
from monai.networks.nets.swin_unetr import SwinTransformer
import torch.nn.functional as F



class vit_classifier(nn.Module):
    
    
    def __init__(self, num_output=32):
        super().__init__()
        

        self.vit = SwinTransformer(
            3, 24, (7, 7, 7), (2, 2, 2), (8, 8, 8, 8), (3, 6, 12, 24), drop_rate=0.1, downsample="mergingv2", use_v2=True,
        )
        
        self.conv_up_0 = nn.Conv3d(24, 48,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv_up_1 = nn.Conv3d(48, 96,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv_up_2 = nn.Conv3d(96, 192,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv_up_3 = nn.Conv3d(192, 384,kernel_size=(3, 3, 3),  # 3x3x3 kernel
                                stride=(2, 2, 2),  # Stride 2 for downsampling
                                padding=1)
        self.conv_up_4 = nn.Conv3d(384, 384,kernel_size=(4, 4, 2),  # 3x3x3 kernel
                                stride=1,  # Stride 2 for downsampling
                                padding=0)
        
        self.output = nn.Linear(384, num_output)
        
        
        
        
        
    def forward(self, x):
        feature0, feature1, feature2, feature3, feature4 = \
            self.vit(x)
        feature1 = nn.functional.relu(self.conv_up_0(feature0) + feature1)
        
        feature2 = nn.functional.relu(self.conv_up_1(feature1) + feature2)

        
        feature3 = nn.functional.relu(self.conv_up_2(feature2) + feature3)

        
        feature4 = nn.functional.relu(self.conv_up_3(feature3) + feature4)

        
        feature = nn.functional.relu(self.conv_up_4(feature4))

        
        feature = feature.flatten(1)
        


        return  self.output(feature)
    
    def forward_encoder(self, x):
        feature0, feature1, feature2, feature3, feature4 = \
            self.vit(x)
        feature1 = nn.functional.relu(self.conv_up_0(feature0) + feature1)
        
        feature2 = nn.functional.relu(self.conv_up_1(feature1) + feature2)

        
        feature3 = nn.functional.relu(self.conv_up_2(feature2) + feature3)

        
        feature4 = nn.functional.relu(self.conv_up_3(feature3) + feature4)

        
        feature = nn.functional.relu(self.conv_up_4(feature4))

        
        feature = feature.flatten(1)
        
        return feature
    
    def inference_zero_shot_cspca_classification(self, x, cspc, non_cspc):
        out_feature = self.forward_encoder(x)
        
        cspc = self.forward_encoder(cspc)
        non_cspc = self.forward_encoder(non_cspc)
        
        cspc = F.cosine_similarity(out_feature, cspc).mean().item()
        non_cspc = F.cosine_similarity(out_feature, non_cspc).mean().item()
        
        
        probability = torch.softmax(torch.tensor([non_cspc, cspc]), dim=-1)
        
        return probability
        
        


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())



class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA parameters (A @ B)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) )
        # B is set to zero
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def apply_lora(model, rank=4, alpha=1.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # replace
            parent = model
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split(".")[-1], LoRALinear(module, rank, alpha))
    return model



if __name__ == '__main__':
    
    
    model = vit_classifier(num_output=6)
    model.eval()
    
    x = torch.rand(1, 3, 128, 128, 64)
    print(model(x))
    print(count_parameters(model, trainable_only=True))
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
        
    for param in model.parameters():
        param.requires_grad = False
    model = apply_lora(model, rank=8, alpha=16)
    print(count_parameters(model, trainable_only=False))
    print(count_parameters(model, trainable_only=True))
    print(model(x))
    
    
    
    
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    
    
    