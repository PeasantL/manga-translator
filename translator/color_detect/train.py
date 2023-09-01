import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from .datasets import ColorDetectionDataset
from .models import get_color_detection_model


def train_model(num_samples=6000, num_workers=7, backgrounds=[], epochs=1000, batch_size=32, learning_rate=0.00,
                weights_path=None, seed=200, train_device=torch.device("cuda:0")):
    dataset = ColorDetectionDataset(generate_target=num_samples, backgrounds=backgrounds, generator_seed=seed,
                                    num_workers=num_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = get_color_detection_model(weights_path=weights_path, device=train_device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    best_loss = 99999999999
    best_epoch = 0
    patience = 20
    patience_count = 0

    best_state = copy.deepcopy(model.state_dict())

    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(dataloader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(dataloader):
                images, results = data
                images = images.type(torch.FloatTensor).to(train_device)
                results = results.type(torch.FloatTensor).to(train_device)

                with torch.cuda.amp.autocast_mode.autocast():
                    outputs: torch.Tensor = model(images)

                    loss = criterion(outputs, results)

                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Calculate the element-wise distance between the arrays
                distance = torch.abs(results - outputs)

                # Set a threshold value for considering the elements as accurate
                threshold = 0.1

                # Calculate the accuracy
                total_accu += (distance <= threshold).float().mean().item()
                total_count += 1
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f'Training | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(dataloader.dataset)):.8f} | ')
                loading_bar.update()

            loading_bar.set_description_str(
                f'Training | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(dataloader.dataset)):.8f} | ')
            loading_bar.close()
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accu = (total_accu / total_count) * 100
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch


    except KeyboardInterrupt:
        print("Stopping training early at user request")

    print("Best epoch at ", best_epoch + 1, "with loss", best_loss)
    model.load_state_dict(best_state)
    return model
