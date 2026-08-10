import time
import torch
import torch.nn as nn
import torch.optim as optim
from configs.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
)
from models.model import EarlyStopping

def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()

        model.train()
        running_train_loss = 0
        for drug_batch, expressions, labels in train_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(labels)
        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for drug_batch, expressions, labels in val_loader:
                drug_batch = drug_batch.to(DEVICE)
                expressions = expressions.to(DEVICE)
                labels = labels.to(DEVICE)
                predictions = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
                loss = criterion(predictions, labels)
                running_val_loss += loss.item() * len(labels)
        val_loss = running_val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        early_stopping.step(val_loss)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    return model, optimizer, train_losses, val_losses


def train_domain_adversarial(model, train_loader, val_loader):
    criterion_response = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    train_response_losses = []
    val_response_losses = []
    val_domain_accuracies = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_response_loss = 0
        for drug_batch, expressions, labels, domain_labels in train_loader:
            drug_batch = drug_batch.to(DEVICE)
            expressions = expressions.to(DEVICE)
            labels = labels.to(DEVICE)
            domain_labels = domain_labels.to(DEVICE)

            optimizer.zero_grad()
            predictions, domain_preds = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
            loss_response = criterion_response(predictions, labels)
            loss_domain = criterion_domain(domain_preds, domain_labels)
            total_loss = loss_response + loss_domain
            total_loss.backward()
            optimizer.step()
            running_response_loss += loss_response.item() * len(labels)
        train_response_loss = running_response_loss / len(train_loader.dataset)

        model.eval()
        running_val_response_loss = 0
        correct_domain_preds = 0
        total_val_samples = 0
        with torch.no_grad():
            for drug_batch, expressions, labels, domain_labels in val_loader:
                drug_batch = drug_batch.to(DEVICE)
                expressions = expressions.to(DEVICE)
                labels = labels.to(DEVICE)
                domain_labels = domain_labels.to(DEVICE)
                predictions, domain_preds = model(drug_batch.x, drug_batch.edge_index, drug_batch.batch, expressions)
                loss_response = criterion_response(predictions, labels)
                running_val_response_loss += loss_response.item() * len(labels)
                predicted_domains = domain_preds.argmax(dim=1)
                correct_domain_preds += (predicted_domains == domain_labels).sum().item()
                total_val_samples += len(labels)
        val_response_loss = running_val_response_loss / len(val_loader.dataset)
        val_domain_accuracy = correct_domain_preds / total_val_samples

        train_response_losses.append(train_response_loss)
        val_response_losses.append(val_response_loss)
        val_domain_accuracies.append(val_domain_accuracy)

        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Response Loss: {train_response_loss:.4f} | Val Response Loss: {val_response_loss:.4f} | Domain Acc: {val_domain_accuracy:.2%}")

        early_stopping.step(val_response_loss)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    return model, optimizer, train_response_losses, val_response_losses, val_domain_accuracies
