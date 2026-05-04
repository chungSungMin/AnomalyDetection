import torch 
import torch.nn.functional as F 


def bilinear_rescaling(feature_maps) : 
    f1_map , f2_map = feature_maps.values()
    
    target_size = f1_map.shape[2:]
    
    f2_map_resized = F.interpolate(
        f2_map,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    
    print(f"1번 feature map의 사이즈 : {f1_map.shape}")
    print(f"2번 feature map의 사이즈 : {f2_map_resized.shape}")
    
    feature_map = torch.concat([f1_map, f2_map_resized], dim=1)
    return feature_map


def apply_local_aggregation(feature_map, p=3) :
    pad_size = p // 2
    aggregated_map = F.avg_pool2d(
        feature_map, 
        kernel_size=p,
        stride=1,
        padding=pad_size
    ) 
    
    return aggregated_map
    
    
def create_patch_vector(feature_map) : 
    B, C, H, W = feature_map.shape 
    
    permuted_map = feature_map.permute(0, 2, 3, 1)
    patch_vectors = permuted_map.reshape(-1, C)
    
    return patch_vectors


def update_memoery_bank(memory_bank_list, feature_vectors) : 
    memory_bank_list.append(feature_vectors.detach().cpu())
    return memory_bank_list