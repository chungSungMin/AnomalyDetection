import json
import torch 
from src.model import parse_args, load_config, get_feature_extractor
from src.utils import bilinear_rescaling, create_patch_vector, update_memoery_bank
from src.coreset import apply_random_projection, get_coreset_indices

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
    
    memory_bank = update_memoery_bank(memory_bank, feature_vectors)
    
    memory_bank = torch.cat(memory_bank, dim=0)
    
    # 위의 내용은 1장 이미지에 대해서 feature vector를 생성하는 코드이다.
    # 여러장의 이미지에 대해서 for문을 반복하고, 최종적으로 memory_bank.cat(memory_backn, dim=0)을 하면 된다.
    
    projected_memory_bank = apply_random_projection(memory_bank, target_dim=128)
    
    coreset_indices = get_coreset_indices(projected_memory_bank, sampling_ratio=0.01)
    
    final_memory_bank = memory_bank[coreset_indices]
    
    print(f"최종적으로 memeory bank 구축 성공 : {final_memory_bank.shape}")
    torch.save(final_memory_bank, "patchore_memory_bank.pt")
    print("dummy data memroy bank 구축 성공")