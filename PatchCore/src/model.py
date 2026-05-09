import os
import argparse
import json
import torch
import torchvision.models as models 
from torchvision.models.feature_extraction import create_feature_extractor

def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore 모델 실험을 위한 세팅.")
    
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    
    parser.add_argument("--backbone", type=str, help="특징벡터를 추출할 백본 모델을 선택하세요 ex)wideresnet50")
    parser.add_argument("--layers", nargs="+", help="추출할 Feature map의 레이어를 선택하세요.")
    parser.add_argument("--patch_size", type=int, help="local smoothing 적용을 위한 패치크기를 설정하세요. ")
    
    return parser.parse_args()


def load_config(args):
    config = {}
    
    if os.path.exists(args.config):
        with open(args.config, "r") as f : 
            config = json.load(f)
    else : 
        print(f"{args.config}를 찾을 수 없습니다.")
    
    # CLI로 파라미터가 넘어오는 경우 덮어쓰기를 진행한다.
    if args.backbone : 
        config["backbone"] = args.backbone
    if args.layers : 
        config["layers"] = args.layers
    if args.patch_size : 
        config["patch_size"] = args.patch_size
        
    return config


def get_feature_extractor(backbone: str, layer_list: list, device) : 
    
   try : 
       base_backbone = models.get_model(backbone, weights="DEFAULT")
   except ValueError : 
       raise ValueError(f"Torchvision에서 지원하지 않거나 오타가 존재하는 이름입니다 : {backbone}")
   
   base_backbone.eval().to(device)
   
   extraction_config = {layer : layer for layer in layer_list}
   
   feature_extractor = create_feature_extractor(
       base_backbone, 
       return_nodes=extraction_config
   )
   
   return feature_extractor
        

