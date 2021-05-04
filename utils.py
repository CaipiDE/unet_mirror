import torch
import os

from PIL import Image
import albumentations as A
import numpy as np


def check_accuracy(loader, model, device):
    running_correct = 0
    running_amount = 0
    running_dice = 0

    model.eval()

    with torch.no_grad():
        for image, ground_truth in loader:
            image = image.to(device)
            ground_truth = ground_truth.to(device).unsqueeze(1)

            prediction = torch.sigmoid(model(image))
            prediction = (prediction > 0.5).float()

            running_correct += (prediction == ground_truth).sum()
            running_amount += torch.numel(prediction)
            running_dice += (2 * (prediction * ground_truth).sum()) / (
                (prediction + ground_truth).sum()
            )

    model.train()

    return running_dice, running_correct / running_amount


def mirrow_extrapolate(image, thickness):

    left_image = get_side(image, thickness, 'l')
    right_image = get_side(image, thickness, 'r')

    horizontal = Image.new(image.mode, size=(image.width + 2*thickness, image.height))
    horizontal.paste(left_image, (0, 0))
    horizontal.paste(image, (left_image.width, 0))
    horizontal.paste(right_image, (left_image.width + image.width, 0))

    tops_image = get_side(horizontal, thickness, 't')
    bottom_image = get_side(horizontal, thickness, 'b')

    full = Image.new(horizontal.mode, size=(horizontal.width, horizontal.height + 2*thickness))
    full.paste(tops_image, (0, 0))
    full.paste(horizontal, (0, tops_image.height))
    full.paste(bottom_image, (0, tops_image.height + horizontal.height))

    return full


def get_side(image, thickness, skyline):
    px = image.load()

    side = []

    if skyline == 'l':
        for i in range(0, image.height):
            for t in range(thickness):
                side.append(px[thickness - t, i])
    elif skyline == 'r':
        for i in range(0, image.height):
            for t in range(thickness):
                side.append(px[image.width - 1 - t, i])
    elif skyline == 't':
        for t in range(thickness):
            for i in range(0, image.width):
                side.append(px[i, thickness - t])
    elif skyline == 'b':
        for t in range(thickness):
            for i in range(0, image.width):
                side.append(px[i, image.height - 1 - t])
    else:
        raise AssertionError("skyline " + skyline + " not existing. Use l, r, t or b instead.")

    if skyline == 'r' or skyline == 'l':
        side_image = Image.new(image.mode, size=(thickness, image.height))
    else:
        side_image = Image.new(image.mode, size=(image.width, thickness))

    side_image.putdata(side)

    return side_image


def multiply_dataset(img_path, mask_path, appedence):
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.6),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.6),
            A.Rotate(always_apply=True, limit=30)
        ]
    )

    # Images and masks name the same
    images = os.listdir(img_path)
    masks = os.listdir(mask_path)

    for image in images:
        if appedence in image:
            os.remove(img_path + image)
    for mask in masks:
        if appedence in mask:
            os.remove(mask_path + mask)

    images = os.listdir(img_path)

    for image_name in images:
        image = np.array(Image.open(img_path + image_name))
        mask = np.array(Image.open(mask_path + image_name))

        augmentations = transform(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']

        Image.fromarray(image).save(img_path + image_name.replace(".tif", appedence + ".tif"))
        Image.fromarray(mask).save(mask_path + image_name.replace(".tif", appedence + ".tif"))
