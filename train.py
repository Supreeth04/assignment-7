import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from datafile import train_loader, test_loader
from models import Model_1, Model_2, Model_3
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, device):
    model = model.to(device)
    summary(model, input_size=(1, 28, 28))

def train_model(model, epochs, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add tqdm progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        train_accuracy = 100. * correct / total
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        # Add tqdm progress bar for validation
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        val_accuracy = 100. * correct / total
        
        print(f'\nEpoch: {epoch+1}/{epochs}')
        print(f'Training Loss: {running_loss/len(train_loader):.3f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
    
    return best_accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example usage
    models = [Model_3()]
    learning_rates = [0.01]
    epochs = [15]
    
    for model, lr, epoch in zip(models, learning_rates, epochs):
        print(f"\nTraining {model.__class__.__name__}")
        print(f"Parameters: {count_parameters(model)}")
        model_summary(model, device)
        best_acc = train_model(model, epochs=epoch, learning_rate=lr)
        print(f"Best accuracy: {best_acc:.2f}%")
