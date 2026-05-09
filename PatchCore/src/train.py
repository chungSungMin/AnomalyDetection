import torch
import logging
from src.model import parse_args, load_config, get_feature_extractor
from src.utils import bilinear_rescaling, create_patch_vector, update_memoery_bank
from src.coreset import apply_random_projection, get_coreset_indices
from src.data.load_data import get_mvtec_dataloader

def train(config):
    device = torch.device(config["device"])
    extractor = get_feature_extractor(config["backbone"], config["layers"], device)
    train_loader, _ = get_mvtec_dataloader(config["root_dir"], config["category"], config["batch_size"], config["num_workers"])
    
    patch_features_list = []
    
    with torch.no_grad():
        for cnt, (images, _) in enumerate(train_loader) : 
            logging.info(f"{cnt} 배치 처리중...")
            images = images.to(device)
            
            features = extractor(images)
            feature_map = bilinear_rescaling(feature_maps=features)
            feature_vectors = create_patch_vector(feature_map, "train")
            memory_bank = update_memoery_bank(patch_features_list, feature_vectors)
            memory_bank = torch.cat(memory_bank, dim=0)
            
        projected_memory_bank = apply_random_projection(memory_bank, target_dim=config["target_dim"])
        coreset_indices = get_coreset_indices(projected_memory_bank, sampling_ratio=config["sampling_ratio"])
            
        final_memory_bank = memory_bank[coreset_indices]
        
        torch.save(final_memory_bank, f"MVtec_{config["category"]}_{config["target_dim"]}_{config["sampling_ratio"]}.pt")
    
    
if __name__ == "__main__" :
    
    memory_bank = []
    
    args = parse_args()
    config = load_config(args)
   
    train(config)