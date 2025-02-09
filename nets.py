import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. Enhanced CNN Model Definition
class GestureCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: Conv + BatchNorm + ReLU
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: Conv + BatchNorm + ReLU
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: Conv + BatchNorm + ReLU
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=f'{data_dir}', transform=train_transform)
    test_dataset = ImageFolder(root=f'{data_dir}', transform=test_transform)
    
    class_counts = torch.bincount(torch.tensor(train_dataset.targets))
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=sampler, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, train_dataset.classes

# 3. Improved OpenCV Integration
class GestureRecognizer:
    def __init__(self, model_path, class_names, device='cpu'):
        self.device = torch.device(device)
        self.model = GestureCNN(len(class_names)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.class_names = class_names
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
        return self.class_names[preds[0]], probs[0].cpu().numpy()

def train_model(model, train_loader, test_loader, n_epochs=25, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True
    )
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(test_loader.dataset)
        val_acc = val_running_corrects.double() / len(test_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_gesture_model.pth')
            print('Saved new best model')

    print(f'Best validation Accuracy: {best_acc:.4f}')
    return model, history

if __name__ == "__main__":
    train_loader, test_loader, class_names = create_dataloaders('data/gestures')
    model = GestureCNN(n_classes=len(class_names))
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=25,
        lr=0.001
    )
    
    torch.save(trained_model.state_dict(), 'gesture_model_final.pth')
