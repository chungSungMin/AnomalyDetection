import torch

def apply_random_projection(memory_bank:torch.Tensor, target_dim:int=128):
    
    N, C = memory_bank.shape
    
    torch.manual_seed(42)
    W = torch.randn(C, target_dim, device=memory_bank.device)
    
    projected_bank = torch.matmul(memory_bank, W)
    return projected_bank


def get_coreset_indices(projected_bank:torch.Tensor, sampling_ratio:int=0.01) -> list : 
    
    N = projected_bank.shape[0]
    coreset_size = int(N * sampling_ratio)
    
    coreset_indices = []
    
    min_distance = torch.full((N,), float("inf"), device=projected_bank.device) # N개의 inf를 생성
    
    current_idx = torch.randint(0, N, (1,)).item() # 0~N 중에서 무작위 수를 뽑아서 1차원을 생성 ( 랜덤 1개의 시작값 )
    
    for i in range(coreset_size):
        coreset_indices.append(current_idx)
        
        center_features = projected_bank[current_idx] # 현재 중심 feature vector
        distances = torch.norm(projected_bank - center_features, dim=1) # 현재 feature vector와 L2-norm을 브로드케스트로 구하게된다.
        
        min_distance = torch.min(min_distance, distances) # distances와 동일한 차원을 갖지만, min_distance와 distance를 비교해서 더작은 값을 갖는 elements로 치환.
        
        current_idx = torch.argmax(min_distance).item() # 치환된 것들 중에서 가장 큰 값을 갖는 index를 가져온다.
    
    print(f"코어셋 인덱스 추출 완료 | 추출된 개수 : {len(coreset_indices)}개")
    return coreset_indices