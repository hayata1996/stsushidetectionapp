import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm
#from torchmetrics import F1Score
import torcheval.metrics.functional as F

def load_model(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()  # Assuming using ResNet18
    num_ftrs = model.fc.in_features
    num_classes = len(os.listdir(data_dir))  # Dynamic based on number of folders
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

def prepare_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def evaluate_model(model, device, dataloader):
    correct = 0
    total = 0
    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)
    
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for confusion matrix
            all_preds = torch.cat((all_preds, predicted))
            all_labels = torch.cat((all_labels, labels))
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
    # Compute the confusion matrix using torcheval
    num_classes = torch.unique(all_labels).numel()  # Number of unique classes
    conf_matrix = F.multiclass_confusion_matrix(all_preds, all_labels, num_classes=num_classes)
    print("Confusion Matrix:")
    print(conf_matrix.numpy())  # Converting to numpy for easier reading if necessary

    
def main():
    model_path = '/Users/kakamuhayata/Desktop/sushidetection/model.pth'
    data_dir = '/Users/kakamuhayata/Desktop/sushidetection/pizza_steak_sushi/test'
    model, device = load_model(model_path, data_dir)
    dataloader = prepare_data(data_dir)
    evaluate_model(model, device, dataloader)

if __name__ == "__main__":
    main()
