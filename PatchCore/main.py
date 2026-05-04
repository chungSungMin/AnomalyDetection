import json
import torch 
from src.model import parse_args, load_config, get_feature_extractor
from src.utils import bilinear_rescaling, create_patch_vector, update_memoery_bank

if __name__ == "__main__" : 
    
    memory_bank = []
    
    args = parse_args()
    config = load_config(args)
    
    # print("=== 현재 config 설정 현황 ===")
    # print(json.dumps(config, indent=4))
    # print("=" * 10)
    
    print("모델을 로드하고 특징추출기를 빌딩하고 있습니다...")
    model = get_feature_extractor(config["backbone"], config["layers"])
    
    dummy_input = torch.randn(1, 3, 244, 244)
    with torch.no_grad():
        features = model(dummy_input)
        
    for layer_name, feature_tensor in features.items():
        print(f"- {layer_name} shape : {feature_tensor.shape}")
        
    feature_map = bilinear_rescaling(feature_maps=features)
    print(f"concat한 후 feature map의 크기 : {feature_map.shape}")
    
    feature_vectors = create_patch_vector(feature_map)
    print(f"feature_vector의 크기 : {feature_vectors.shape}")
    
    update_memoery_bank(memory_bank, feature_vectors)
    
    # 위의 내용은 1장 이미지에 대해서 feature vector를 생성하는 코드이다.
    # 여러장의 이미지에 대해서 for문을 반복하고, 최종적으로 memory_bank.cat(memory_backn, dim=0)을 하면 된다.