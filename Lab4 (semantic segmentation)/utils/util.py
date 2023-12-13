import random
import matplotlib.pyplot as plt
from utils.mask_transform import multiclass_to_rgb


def plot_samples(dataset, n):
    random_ids = [random.randint(0, len(dataset) - 1) for _ in range(n)]

    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    for i, idx in enumerate(random_ids):
        img_t, mask_t, class_tensor = dataset[idx]

        img = img_t.permute(1, 2, 0).numpy()
        mask = mask_t.permute(1, 2, 0).numpy()

        row = i
        col = 0

        ax = axes[row, col]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Sample {idx} - Image')

        col = 1

        ax = axes[row, col]
        ax.imshow(mask)
        ax.axis('off')
        ax.set_title(f'Sample {idx} - Mask')

    plt.tight_layout()
    plt.show()


def plot_multiclass(class_tensor, class_list):
    rows, cols = 4, 6
    fig = plt.figure(figsize=(15, 10))

    for i in range(24):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(class_tensor[i], cmap='viridis')
        ax.set_title(class_list[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    return '{:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))


def single_test(model, ld, class_rgb):
    batch = next(iter(ld))
    img, mask, classes = batch
    model.eval()
    output = model(img.to('mps'))
    i = img[0]
    m = mask[0]
    out = output[0]

    o = multiclass_to_rgb(out, class_rgb=class_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(i.to('cpu').permute(1, 2, 0).numpy())
    axes[0].set_title("Image")
    axes[0].axis('off')
    axes[1].imshow(m.to('cpu').permute(1, 2, 0).numpy())
    axes[1].set_title("True Mask")
    axes[1].axis('off')
    axes[2].imshow(o.permute(1, 2, 0).to('cpu').numpy())
    axes[2].set_title('{} Mask'.format(model.__class__.__name__))
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def print_parameters_count(model):
    print('{} parameters count: {}'.format(model.__class__.__name__, count_parameters(model)))


def triple_test(ld, class_rgb, model1,model2,model3):
    n = 4
    fig, axes = plt.subplots(n, 5, figsize=(20, 15))
    for idx in range(n):
        batch = next(iter(ld))
        imgs, masks, classes = batch
        model1.eval()
        model2.eval()
        model3.eval()

        imgs = imgs.to('mps')
        o1 = model1(imgs)
        o2 = model2(imgs)
        o3 = model3(imgs)

        i = imgs[0]
        m = masks[0]
        o1 = multiclass_to_rgb(o1[0], class_rgb=class_rgb)
        o2 = multiclass_to_rgb(o2[0], class_rgb=class_rgb)
        o3 = multiclass_to_rgb(o3[0], class_rgb=class_rgb)

        axes[idx, 0].imshow(i.to('cpu').permute(1, 2, 0).numpy())
        axes[idx, 0].set_title("Image")
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(m.to('cpu').permute(1, 2, 0).numpy())
        axes[idx, 1].set_title("Mask")
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(o1.permute(1, 2, 0).to('cpu').numpy())
        axes[idx, 2].set_title('{}'.format(model1.__class__.__name__))
        axes[idx, 2].axis('off')
        axes[idx, 3].imshow(o2.permute(1, 2, 0).to('cpu').numpy())
        axes[idx, 3].set_title('{}'.format(model2.__class__.__name__))
        axes[idx, 3].axis('off')
        axes[idx, 4].imshow(o3.permute(1, 2, 0).to('cpu').numpy())
        axes[idx, 4].set_title('{}'.format(model3.__class__.__name__))
        axes[idx, 4].axis('off')

    plt.tight_layout()
    plt.show()

