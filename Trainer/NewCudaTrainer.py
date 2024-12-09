import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

#res and load csv into df
IMG_RESOLUTION = 256
data_file = 'test.csv'
data_df = pd.read_csv(data_file)

#assign all columns a numerical code and fill any missing values with -1
categorical_columns = ['region', 'sub-region', 'drive_side', 'climate', 'soil', 'land_cover']
data_df[categorical_columns] = data_df[categorical_columns].apply(lambda x: x.astype('category').cat.codes).fillna(-1)

#gets the mapping data from country/city_mapping and makes them a dict
country_to_index, city_to_index = {}, {}
index_to_country, index_to_city = {}, {}
def get_or_add_mapping(mapping, reverse_mapping, name):
    if name not in mapping:
        idx = len(mapping)
        mapping[name], reverse_mapping[idx] = idx, name
    return mapping[name]

#adds any missing entries if necessary, then puts those dicts into a dataframe
data_df['country_index'] = data_df['country'].apply(lambda x: get_or_add_mapping(country_to_index, index_to_country, x))
data_df['city_index'] = data_df['city'].apply(lambda x: get_or_add_mapping(city_to_index, index_to_city, x))

# Dataset class
class ImageCountryDataset(Dataset):
    def __init__(self, image_dir, data_df, img_size, start_idx=10001, transform=None):
        self.image_dir = image_dir
        self.data_df = data_df.iloc[start_idx:]
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        actual_idx = idx + 10001
        country_index = self.data_df.iloc[idx]['country_index']
        city_index = self.data_df.iloc[idx]['city_index']
        additional_features = torch.tensor(self.data_df.iloc[idx][categorical_columns].astype(float).values, dtype=torch.float32)
        
        img_path = os.path.join(self.image_dir, f"{actual_idx}.png")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Missing image at {img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        if self.transform:
            image = self.transform(image)

        return image, additional_features, (country_index, city_index)

# Transformations and Dataloader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_RESOLUTION, IMG_RESOLUTION)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Collate function to handle None values
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    images, additional_features, labels = zip(*batch)
    country_labels, city_labels = zip(*labels)
    return torch.stack(images), torch.stack(additional_features), (torch.tensor(country_labels), torch.tensor(city_labels))

dataset = ImageCountryDataset("Newdataset", data_df, IMG_RESOLUTION, transform=transform)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define model
class MultiInputModel(nn.Module):
    def __init__(self, num_countries, num_cities, additional_features_dim):
        super().__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  
        
        self.additional_fc = nn.Sequential(
            nn.Linear(additional_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2163227635),  # Add dropout for regularization
            nn.Linear(512, 64)
        )
        
        self.final_fc = nn.Linear(in_features + 64, num_countries + num_cities)

    def forward(self, image, additional_features):
        img_features = self.resnet(image)
        additional_features = self.additional_fc(additional_features)
        combined_features = torch.cat((img_features, additional_features), dim=1)
        return self.final_fc(combined_features)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiInputModel(len(country_to_index), len(city_to_index), len(categorical_columns)).to(device)

# Training configuration
optimizer = optim.Adam(model.parameters(), lr=0.0002762387809, weight_decay=0.0001326445686)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
country_loss_fn = city_loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs, patience = 100, 5  # Increased patience
best_val_loss, no_improvement_epochs = float('inf'), 0
 
for epoch in range(num_epochs):
    start_time = time.time()  # Start timer for epoch
    
    model.train()
    running_loss = 0.0
    for batch_idx, (images, additional_features, (country_labels, city_labels)) in enumerate(train_loader):
        images, additional_features = images.to(device).float(), additional_features.to(device)
        country_labels, city_labels = country_labels.to(device).long(), city_labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images, additional_features)
        country_loss = country_loss_fn(outputs[:, :len(country_to_index)], country_labels)
        city_loss = city_loss_fn(outputs[:, len(country_to_index):], city_labels)
        loss = country_loss + city_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, additional_features, (country_labels, city_labels) in val_loader:
            images, additional_features = images.to(device).float(), additional_features.to(device)
            country_labels, city_labels = country_labels.to(device).long(), city_labels.to(device).long()
            outputs = model(images, additional_features)
            val_loss += (country_loss_fn(outputs[:, :len(country_to_index)], country_labels) + 
                         city_loss_fn(outputs[:, len(country_to_index):], city_labels)).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss, no_improvement_epochs = avg_val_loss, 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print("Early stopping triggered.")
            break
    
    scheduler.step()
    
    # Calculate and print epoch duration
    epoch_duration = time.time() - start_time
    print(f"Epoch Duration: {epoch_duration:.2f} seconds")

print("\nTraining complete.")
