import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import wandb

from dataset import VoxelDataset

class CNN3D(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super(CNN3D, self).__init__()
        
        self.lr = lr
        self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 5, 5))
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 5, 5))
        self.bn2 = nn.BatchNorm3d(16)
        # self.conv3 = nn.Conv3d(16, 32, kernel_size=(2, 5, 5))
        # self.bn3 = nn.BatchNorm3d(32)
        self.pool = nn.MaxPool3d(kernel_size=3)
        self.fc1 = nn.Linear(16 * 1 * 40 * 40, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(0.2)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        
        self.val_precision = torchmetrics.classification.Precision(task="binary")
        self.val_recall = torchmetrics.classification.Recall(task="binary")
        self.val_f1 = torchmetrics.classification.F1Score(task="binary")
        
        self.train_loss = torchmetrics.aggregation.MeanMetric()
        self.val_loss = torchmetrics.aggregation.MeanMetric()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # print("Shape after conv1:", x.shape)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # print("Shape after conv2:", x.shape)
        # x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # print("Shape after conv3:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.log("train_acc_epoch", acc)
        self.log("train_loss", self.train_loss.compute())
        self.train_acc.reset()
        self.train_loss.reset()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels)
        # self.log('train_loss', loss)
        self.train_loss.update(loss)
        
        _, preds = torch.max(outputs, 1)
        self.train_acc.update(preds, labels)
        
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log('valid_acc_epoch', acc)
        self.log("val_loss", self.val_loss.compute())
        
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        
        print(f"Validation Accuracy: {acc}")
        print(f"Validation Precision: {precision}")
        print(f"Validation Recall: {recall}")
        print(f"Validation F1 Score: {f1}")
        print(f"Validation Loss: {self.val_loss.compute()}")
        
        self.log('valid_acc_epoch', acc)
        self.log('val_precision_epoch', precision)
        self.log('val_recall_epoch', recall)
        self.log('val_f1_epoch', f1)
        self.log("val_loss", self.val_loss.compute())

        # Reset all metrics for next epoch
        self.val_acc.reset()
        self.val_loss.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        # print("OUTPUTS:", outputs)
        # print("PREDS:", preds)
        # print("LABELS:", labels)
        
        self.val_acc.update(preds, labels)
        self.val_loss.update(loss)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)
        
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        
        # print("val step", labels, outputs)
        # precision = precision_score(labels, preds)
        # recall = recall_score(labels, preds)
        # f1 = f1_score(labels, preds)
        # self.log('val_loss', loss)
        # self.log('val_accuracy', accuracy)
        # self.log('val_precision', precision)
        # self.log('val_recall', recall)
        # self.log('val_f1', f1)
        
        # print(f"Validation Loss: {loss.item()}")
        # print(f"Validation Accuracy: {accuracy}")
        # print(f"Validation Precision: {precision}")
        # print(f"Validation Recall: {recall}")
        # print(f"Validation F1 Score: {f1}")
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        
class CNN2D(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super(CNN2D, self).__init__()
        print("loading model")
        
        self.lr = lr
        
        self.conv1 = nn.Conv2d(24, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        
        print("model loaded")

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.log("train_acc_epoch", acc)
        self.train_acc.reset()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        
        _, preds = torch.max(outputs, 1)
        self.train_acc.update(preds, labels)
        
        return loss
    
    def on_validation_epoch_end(self):
        print("finishing validation epoch")
        acc = self.val_acc.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        
        print(f"Validation Accuracy: {acc}")
        print(f"Validation Precision: {precision}")
        print(f"Validation Recall: {recall}")
        print(f"Validation F1 Score: {f1}")
        print(f"Validation Loss: {self.val_loss.compute()}")
        
        self.log('valid_acc_epoch', acc)
        self.log('val_precision_epoch', precision)
        self.log('val_recall_epoch', recall)
        self.log('val_f1_epoch', f1)
        self.log("val_loss", self.val_loss.compute())

        # Reset all metrics for next epoch
        self.val_acc.reset()
        self.val_loss.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        print("outputs", outputs.shape, labels)
        loss = self.criterion(outputs, labels)
        print("val loss", loss)
        _, preds = torch.max(outputs, 1)
        self.val_acc.update(preds, labels)
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        
        # print("val step", labels, outputs)
        # precision = precision_score(labels, preds)
        # recall = recall_score(labels, preds)
        # f1 = f1_score(labels, preds)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        # self.log('val_precision', precision)
        # self.log('val_recall', recall)
        # self.log('val_f1', f1)
        
        # print(f"Validation Loss: {loss.item()}")
        # print(f"Validation Accuracy: {accuracy}")
        # print(f"Validation Precision: {precision}")
        # print(f"Validation Recall: {recall}")
        # print(f"Validation F1 Score: {f1}")
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=5)

# if __name__ == "__main__":
#     model = CNN3D()
    
#     model.eval()
#     with torch.no_grad():
#         x = torch.randn(10, 3, 8, 128, 128)  # Dummy batch of 10 samples
#         output = model(x)
#         probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
#         preds = torch.argmax(probs, dim=1)  # Get class predictions

#     print("Predicted class counts:", preds)

# Example usage
if __name__ == "__main__":
     
    config = {
        "bs": 32,
        "lr": 5e-6,
        "comments": "adding cspca synthetic samples",
        "synthetic_cspca_samples": 0
    }
    
    wandb.init(project="cspca-classifier-3", config=config)    
    wandb_logger = WandbLogger(log_model="all")

    print("Starting training...")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images if necessary
    ])
    # Assuming you have a dataset and dataloader
    train_dataset = VoxelDataset(
        chunks_csv_path="/home/sr2369/inpainting/gen_train.csv",
        data_type="train",
        base_path="/home/sr2369/inpainting/max_area_3d_voxel",
        transform=transform,
        return_meta=False,
        normalize=True,
        sample_path="/home/sr2369/inpainting/inference/only_cancer_cond",
        use_samples=True,
        num_samples_to_use=0
    )
    
    print(f"Training Dataset Len: {len(train_dataset)}")
    
    val_dataset = VoxelDataset(
        chunks_csv_path="/home/sr2369/inpainting/gen_val.csv",
        data_type="val",
        base_path="/home/sr2369/inpainting/max_area_3d_voxel",
        transform=transform,
        return_meta=False,
        normalize=True
    )
    
    model = CNN3D(lr=config['lr'])
    
    print("Model loaded", model)
    
    # out = model(torch.randn(1, 24, 128, 128))
    # print("output", out.shape)
    
    data_module = DataModule(train_dataset, val_dataset, batch_size=config["bs"])
    
    print("Module laoded")
    
    trainer = pl.Trainer(max_epochs=25, accelerator="gpu", logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data_module)
