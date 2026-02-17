import numpy as np
from fire import Fire
import cv2


def main(n_data_samples=-1,
         digit=tuple(range(10)),
         img_size=28,
         flatten_imgs=True,
         test=False):
    # Load data:
    # Raw data was downloaded here (and store with mnist_ prefix):
    # https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/master/data/MNIST
    mnist = np.load(f"mnist_{'test_' if test else 'train_'}images.npy")
    mnist_labels = np.load(f"mnist_{'test_' if test else 'train_'}labels.npy")
    assert mnist.shape[0] == mnist_labels.size
    n_pixels = mnist.shape[1]
    mnist_size = round(np.sqrt(n_pixels))
    assert mnist_size ** 2 == n_pixels

    if test:
        print("TEST DATA")

    # Re-shape data:
    mnist = mnist.reshape(-1, mnist_size, mnist_size)
    mnist_labels = mnist_labels.flatten()
    print(mnist.shape)
    print(mnist_labels.shape)

    # Filter digits:
    if isinstance(digit, int):  # single digit, otherwise, iterable with multiple desired digits
        digit = [digit]
    digit = np.asarray(digit)
    keep_indices = np.vectorize(lambda x: x in digit)(mnist_labels)
    mnist = mnist[keep_indices]
    mnist_labels = mnist_labels[keep_indices]
    print(f"{len(mnist)} samples with digits: {digit.squeeze()}")

    # Keep only specified number of samples:
    if n_data_samples is None or n_data_samples < 0:
        n_data_samples = len(mnist)
    mnist = mnist[:n_data_samples]
    print(len(mnist))

    # Downscale images to desired size (with proper interpolation handled by opencv):
    imgs_scaled = np.empty((n_data_samples, img_size, img_size))
    for i, img in enumerate(mnist):
        imgs_scaled[i] = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    print("Sample\n", imgs_scaled[0], mnist_labels[0])
    print(imgs_scaled.shape)

    # Flatten images again before saving if desired:
    if flatten_imgs:
        imgs_scaled = imgs_scaled.reshape(n_data_samples, -1)
        print("Flattened shape:", imgs_scaled.shape)

    # Store final images:
    digits_str = '_'.join(map(str, sorted(digit)))
    save_path = f"mnist_{digits_str}_{img_size}x{img_size}_N_{n_data_samples}{'_TEST' if test else ''}"
    np.savez_compressed(save_path, X=imgs_scaled, y=mnist_labels[:n_data_samples])


if __name__ == '__main__':
    Fire(main)
