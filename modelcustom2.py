import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from glob import glob
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_labels = []

        for i, disease in enumerate(sorted(os.listdir(root_dir))):
            disease_dir = os.path.join(root_dir, disease)
            if os.path.isdir(disease_dir):
                for img_file in glob(os.path.join(disease_dir, '*.jpg')):
                    self.image_labels.append((img_file, i))
        #config the correct labels: i for each images.0-7. Put them as a tupple in image_labels.

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def main():
    data_dir = '/Users/kakamuhayata/Desktop/sushidetection/pizza_steak_sushi/train'
    model_save_path = 'model.pth'
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(os.listdir(data_dir)))
    model.fc.requires_grad = True
    model.to(device)

    train_transform = prepare_data()
    dataset = CustomDataset(data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs
    }, model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    main()