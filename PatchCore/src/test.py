import torch
import logging
import torch.nn.functional as F 
from src.model import parse_args, load_config, get_feature_extractor
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import bilinear_rescaling, create_patch_vector
from src.data.load_data import get_mvtec_dataloader

def train(config):
    device = torch.device(config["device"])
    extractor = get_feature_extractor(config["backbone"], config["layers"], device)
    _, test_loader = get_mvtec_dataloader(config["root_dir"], config["category"], config["batch_size"], config["num_workers"])
    memory_bank = torch.load(f"MVtec_{config["category"]}_{config["target_dim"]}_{config["sampling_ratio"]}.pt").to(device)
    
    all_image_scores, all_score_maps, all_labels = [], [], []
    
    with torch.no_grad():
        for cnt, (images, labels) in enumerate(test_loader) : 
            logging.info(f"{cnt} 배치 처리중...")
            images = images.to(device)
            
            features = extractor(images)
            feature_map = bilinear_rescaling(feature_maps=features)
            B, C, H, W = feature_map.shape
            feature_vectors = create_patch_vector(feature_map, "eval") # (B, -1, C)
            '''
            feature_vectors = [B, H*W, C]
            memory_bank = [M, C] # M은 대표 vector의 개수
            '''
            dist = torch.cdist(feature_vectors, memory_bank.unsqueeze(0).expand(B, -1, -1))
            '''
            feature_vectors = [B, H*W, C]
            memory_bank = [B, M, C]
            결국 C 기준으로 거리를 계산 하기 때문에 dist = [B, H*W, M]이 되게된다. 그래서 2차원 행렬에서 각 H*W행에 매칭되는 M열이 거리가 된다.
            '''
            
            patch_scores, _ = dist.min(dim=2) # 가장 가까운 memorybank와의 거리들  [ B, H*W ]
            score_maps = patch_scores.reshape(B, H, W) # 각 이미지에서 각 픽셀 위치마다 가장 가까운 거리 저장.
            image_scores = patch_scores.max(dim=1).values # feature vector 중 가장 이상한 점수 가져오기.
            
            score_maps_up = F.interpolate(
                score_maps.unsqueeze(1), size=(244, 244),
                mode="bilinear", align_corners=False
            ).squeeze(1)

            all_image_scores.append(image_scores.cpu())
            all_score_maps.append(score_maps_up.cpu())
            all_labels.append(labels)
            
        all_image_scores = torch.cat(all_image_scores)
        all_score_maps = torch.cat(all_score_maps)
        all_labels = torch.cat(all_labels)
        
        
        img_auroc = roc_auc_score(all_labels.numpy(), all_image_scores.numpy())
        img_ap    = average_precision_score(all_labels.numpy(), all_image_scores.numpy())

        print(f"Image-level AUROC: {img_auroc:.4f}")
        print(f"Image-level AP   : {img_ap:.4f}")

if __name__ == "__main__" :
    
    memory_bank = []
    
    args = parse_args()
    config = load_config(args)
   
    train(config)