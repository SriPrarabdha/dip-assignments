import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional

from .otsu_ic import optimal_otsu_ic

class ConnectedComponentExtractor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_img = np.array(Image.open(image_path))

        otsu_inter_class = optimal_otsu_ic(image_path)
        self.thres = otsu_inter_class.get_threshold(debug=True)

        self.binary_img = (self.original_img < self.thres).astype(np.uint8)

    def _connected_components(self, binary_img):
        """Label connected components with 8-connectivity."""
        h, w = binary_img.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label = 0
        areas = {}

        # 8-neighbour relative positions
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        for i in range(h):
            for j in range(w):
                if binary_img[i, j] == 1 and labels[i, j] == 0:
                    label += 1
                    q = deque([(i, j)])
                    labels[i, j] = label
                    area = 0

                    while q:
                        x, y = q.popleft()
                        area += 1
                        for dx, dy in neighbors:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < h and 0 <= ny < w and
                                binary_img[nx, ny] == 1 and labels[nx, ny] == 0):
                                labels[nx, ny] = label
                                q.append((nx, ny))

                    areas[label] = area
        return labels, areas

    def extract_all_components(self):
        labels, areas = self._connected_components(self.binary_img)
        num_labels = len(areas)

        rng = np.random.default_rng(42)
        colors = rng.integers(50, 255, size=(num_labels + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] 

        # color map image
        h, w = labels.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for lbl in range(1, num_labels + 1):
            color_img[labels == lbl] = colors[lbl]

        if self.original_img.ndim == 2:  
            highlighted = np.stack([self.original_img]*3, axis=-1).copy()
        else: 
            highlighted = self.original_img.copy()

        mask = labels > 0
        highlighted[mask] = color_img[mask]

        return color_img, highlighted


    def plot_results(self, save_dir:Optional[str] = None):
        color_img, highlighted = self.extract_all_components()

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].imshow(self.original_img, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(highlighted)
        axes[1].set_title("All Components")
        axes[1].axis("off")
        if(save_dir): plt.savefig(f"{save_dir}/task4_cce.png",
                                  bbox_inches='tight', dpi=150)

        plt.show()


