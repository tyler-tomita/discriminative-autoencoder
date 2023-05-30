import torch
import time
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dae(model, dataloaders, criterion_autoencoder, criterion_discriminator, optimizer, lambda_autoencoder, num_epochs=10):
    # trains a discrminative autoencoder
    # discriminator weight is the weight w in loss = (1-w)*reconstruction_error + w*negative_log_likelihood

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    dataset_sizes = {}
    for phase in ('train', 'val'):
        dataset_sizes[phase] = len(dataloaders[phase].dataset)

    # Get initial validation loss
    # Iterate over data.
    phase = 'val'
    model.eval()
    running_loss = 0.0
    running_discriminator_loss = 0.0
    running_classification_error = 0.0
    running_reconstruction_error = 0.0

    total_losses = {}
    reconstruction_errors = {}
    discriminator_losses = {}
    classification_errors = {}
    for phase in ('train', 'val'):
        if phase == 'train':
            tensor_length = num_epochs
        else:
            tensor_length = num_epochs + 1
        total_losses[phase] = torch.zeros(tensor_length, dtype=torch.float)
        reconstruction_errors[phase] = torch.zeros(tensor_length, dtype=torch.float)
        discriminator_losses[phase] = torch.zeros(tensor_length, dtype=torch.float)
        classification_errors[phase] = torch.zeros(tensor_length, dtype=torch.float)

    # compute initial validation performance
    for inputs, labels in dataloaders[phase]:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs_discriminator, outputs_autoencoder = model(inputs)
            loss_autoencoder = criterion_autoencoder(outputs_autoencoder.to(device), inputs).view(inputs.size(0), -1).mean(dim=1).sum()
            loss_autoencoder = loss_autoencoder / inputs.size(0)
            loss_discriminator = criterion_discriminator(outputs_discriminator.squeeze().to(device), labels)
            loss = loss_discriminator + lambda_autoencoder*loss_autoencoder
            error_rate = (outputs_discriminator.topk(1, dim=1)[1].squeeze() != labels).sum()
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_discriminator_loss += loss_discriminator.item() * inputs.size(0)
        running_reconstruction_error += loss_autoencoder.item() * inputs.size(0)
        running_classification_error += error_rate

    total_losses[phase][0] = running_loss / dataset_sizes[phase]
    reconstruction_errors[phase][0] = running_reconstruction_error / dataset_sizes[phase]
    discriminator_losses[phase][0] = running_discriminator_loss / dataset_sizes[phase]
    classification_errors[phase][0] = running_classification_error / dataset_sizes[phase]

    best_loss = total_losses[phase][0]
    best_discriminator_loss = discriminator_losses[phase][0]
    best_reconstruction_error = reconstruction_errors[phase][0]
    best_error = classification_errors[phase][0]

    print(f'{phase} Loss: {total_losses[phase][0]:.4f}')
    print(f'{phase} Reconstruction Error: {reconstruction_errors[phase][0]:.4f}')
    print(f'{phase} Discriminator Loss: {discriminator_losses[phase][0]:.4f}')
    print(f'{phase} Classification Error: {classification_errors[phase][0]:.4f}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_discriminator_loss = 0.0
            running_reconstruction_error = 0.0
            running_classification_error = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_discriminator, outputs_autoencoder = model(inputs)
                    loss_autoencoder = criterion_autoencoder(outputs_autoencoder.to(device), inputs).view(inputs.size(0), -1).mean(dim=1).sum()
                    loss_autoencoder = loss_autoencoder / inputs.size(0)
                    loss_discriminator = criterion_discriminator(outputs_discriminator.squeeze().to(device), labels)
                    loss = loss_discriminator + lambda_autoencoder*loss_autoencoder
                    error_rate = (outputs_discriminator.topk(1, dim=1)[1].squeeze() != labels).sum()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_discriminator_loss += loss_discriminator.item() * inputs.size(0)
                running_reconstruction_error += loss_autoencoder.item() * inputs.size(0)
                running_classification_error += error_rate

            if phase == 'train':
                epoch_idx = epoch
            else:
                epoch_idx = epoch + 1

            total_losses[phase][epoch_idx] = running_loss / dataset_sizes[phase]
            reconstruction_errors[phase][epoch_idx] = running_reconstruction_error / dataset_sizes[phase]
            discriminator_losses[phase][epoch_idx] = running_discriminator_loss / dataset_sizes[phase]
            classification_errors[phase][epoch_idx] = running_classification_error / dataset_sizes[phase]

            print(f'{phase} Loss: {total_losses[phase][epoch_idx]:.4f}')
            print(f'{phase} Reconstruction Error: {reconstruction_errors[phase][epoch_idx]:.4f}')
            print(f'{phase} Discriminator Loss: {discriminator_losses[phase][epoch_idx]:.4f}')
            print(f'{phase} Classification Error: {classification_errors[phase][epoch_idx]:.4f}')

            # deep copy the model
            if phase == 'val' and discriminator_losses[phase][epoch_idx] < best_discriminator_loss:
                best_loss = running_loss / dataset_sizes[phase]
                best_discriminator_loss = discriminator_losses[phase][epoch_idx]
                best_error = classification_errors[phase][epoch_idx]
                best_reconstruction_error = reconstruction_errors[phase][epoch_idx]
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Val Total Loss: {best_loss:4f}')
    print(f'Val Discriminator Loss: {best_discriminator_loss:4f}')
    print(f'Val Error: {best_error:4f}')
    print(f'Reconstruction Error: {best_reconstruction_error:.4f}')
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, total_losses, reconstruction_errors, discriminator_losses, classification_errors
