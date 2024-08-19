import torch
from torchvision import transforms


def custom_collate_fn(batch):
    images, targets = zip(*batch)

    # Define a transformation to convert images to tensor
    transform_to_tensor = transforms.ToTensor()

    # Convert images to tensor
    images = torch.stack([transform_to_tensor(img) for img in images], dim=0)

    # Find the maximum length of the target tensors
    max_target_len = max(t.size(0) for t in targets)

    # Pad targets to the same length
    targets_padded = [
        torch.cat([
            t,
            torch.zeros((max_target_len - t.size(0), t.size(1)),
                        dtype=t.dtype, device=t.device)
        ], dim=0) for t in targets
    ]

    # Convert list of padded targets to a tensor
    targets_padded = torch.stack(targets_padded, dim=0)

    return images, targets_padded
