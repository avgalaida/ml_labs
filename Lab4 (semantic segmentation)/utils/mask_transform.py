import torch


def rgb_to_multiclass(mask, class_rgb):
    with torch.no_grad():
        num_classes = len(class_rgb)
        height, width = mask.size(1), mask.size(2)
        mulclass_tensor = torch.zeros(num_classes, height, width)

        for class_idx, (class_name, rgb_color) in enumerate(class_rgb.items()):
            rgb = [int(x) for x in rgb_color.split(',')]
            r_mask = torch.where(mask[0] == rgb[0] / 255, 1, 0)
            g_mask = torch.where(mask[1] == rgb[1] / 255, 1, 0)
            b_mask = torch.where(mask[2] == rgb[2] / 255, 1, 0)
            class_mask = r_mask * g_mask * b_mask

            mulclass_tensor[class_idx] = class_mask

    return mulclass_tensor


def multiclass_to_rgb(class_tensor, class_rgb):
    with torch.no_grad():
        class_tensor = class_tensor.to(torch.device('mps'))
        num_classes = class_tensor.size(0)
        height, width = class_tensor.size(1), class_tensor.size(2)

        rgb_mask = torch.zeros(3, height, width, dtype=torch.uint8).to(torch.device('mps'))

        for class_idx in range(num_classes):
            class_mask = class_tensor[class_idx]

            class_name = list(class_rgb.keys())[class_idx]
            rgb_color = [int(x) for x in class_rgb[class_name].split(',')]

            r_channel = torch.where(class_mask > 0, rgb_color[0], 0)
            g_channel = torch.where(class_mask > 0, rgb_color[1], 0)
            b_channel = torch.where(class_mask > 0, rgb_color[2], 0)

            rgb_mask[0] += r_channel
            rgb_mask[1] += g_channel
            rgb_mask[2] += b_channel

    return rgb_mask
