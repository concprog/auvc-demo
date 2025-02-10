import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from tqdm import tqdm

class GestureCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels.
            # For 128x128 input, after conv (padding=1) we have 128x128,
            # then MaxPool2d(2) yields 32 x 64 x 64.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 2: 32 -> 64 channels.
            # Input size 64x64 becomes 32x32 after pooling.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128 channels.
            # 32x32 becomes 16x16 after pooling.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        # For 128x128 input, after three poolings (dividing each dimension by 8) the feature map is 16x16.
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
        

def create_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(root=data_dir, transform=train_transform)
    test_dataset = ImageFolder(root=data_dir, transform=test_transform)

    targets = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader, train_dataset.classes

class GestureRecognizer:
    def __init__(self, model_path, class_names, device='cpu', use_fp16=False):
        
        self.device = torch.device(device)
        self.class_names = class_names
        self.use_fp16 = use_fp16 and (self.device.type != 'cpu')
        
        # Initialize and load the model.
        self.model = GestureCNN(len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        if self.use_fp16:
            self.model.half() 
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
  
        if self.use_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
        return self.class_names[preds[0]], probs[0].cpu().numpy()

def train_model(model, train_loader, test_loader, n_epochs=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     patience=3, factor=0.5, verbose=True)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        # Training phase
        model.train()
        total_loss, total_correct = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / len(test_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)
        print(f"  Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}\n")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.half()               
            torch.save(model.state_dict(), 'best_gesture_model_fp16.pth')
            model.float()              
            print("Saved new best model in FP16.")
    
    print(f"Best validation Accuracy: {best_acc:.4f}")

    model.half()
    torch.save(model.state_dict(), 'gesture_model_final_fp16.pth')
    return model, history


if __name__ == '__main__':
    train_loader, test_loader, class_names = create_dataloaders('data/gestures')
    model = GestureCNN(n_classes=len(class_names))
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=50,
        lr=0.001
    )
    
    torch.save(trained_model.state_dict(), 'gesture_model_final.pth')
