import os 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTecDataset(Dataset) : 
    def __init__ (self, root_dir, category, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.image_paths = [] 
        self.labels = []
        
        phase = 'train' if is_train else 'test'
        data_dir = os.path.join(root_dir, category, phase)
        
        for defect_type in os.listdir(data_dir) : 
            defect_dir = os.path.join(data_dir, defect_type)
            if not os.path.isdir(defect_dir) : 
                continue 
            
            label = 0 if defect_type == "good" else 1
            
            for img_name in os.listdir(defect_dir):
                if img_name.endswith((".png", ".jpg", ".jpeg")) : 
                    self.image_paths.append(os.path.join(defect_dir, img_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform : 
            image = self.transform(image)
        
        return image, label
            

def get_mvtec_dataloader(root_dir, category, batch_size=32, num_workers=1) : 
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    train_dataset = MVTecDataset(root_dir, category, True, transform)
    test_dataset = MVTecDataset(root_dir, category, False, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader