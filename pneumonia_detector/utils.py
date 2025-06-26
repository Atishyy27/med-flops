import matplotlib.pyplot as plt
import torch

def visualize_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
