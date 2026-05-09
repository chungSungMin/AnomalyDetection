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
    
    
def create_patch_vector(feature_map, is_train) : 
    B, C, _, _ = feature_map.shape 
    
    if is_train == "train" : 
        permuted_map = feature_map.permute(0, 2, 3, 1).reshape(-1, C)    
        return permuted_map
    else : 
        permuted_map = feature_map.permute(0, 2, 3, 1).reshape(B, -1, C)
        return permuted_map
        


def update_memoery_bank(memory_bank_list, feature_vectors) : 
    memory_bank_list.append(feature_vectors.detach().cpu())
    return memory_bank_list