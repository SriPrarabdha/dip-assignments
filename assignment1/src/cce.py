import numpy as np
from collections import deque
from PIL import Image
from .otsu_ic import optimal_otsu_ic
import matplotlib.pyplot as plt

class ConnectedComponents:
    def __init__(self, image_path:str):
        otsu_inter_class = optimal_otsu_ic(image_path, 0)
        thres = otsu_inter_class.get_threshold()
        
        img_arr = np.array(Image.open(image_path), dtype=np.uint8)
        
        self.binary_image = (img_arr > thres).astype(np.uint8)
        self.h, self.w = self.binary_image.shape
        
        self.labels = np.zeros_like(self.binary_image, dtype=np.uint32)
        self.current_label = 0

        # 8-neighbourhood
        self.neigh = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

    def bfs(self, x, y):
        q = deque()
        q.append((x, y))
        self.current_label += 1
        self.labels[x, y] = self.current_label
        size = 1

        while q:
            cx, cy = q.popleft()
            for dx, dy in self.neigh:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.h) and (0 <= ny < self.w):
                    if self.binary_image[nx, ny] == 1 and self.labels[nx, ny] == 0:
                        self.labels[nx, ny] = self.current_label
                        q.append((nx, ny))
                        size += 1
        return size

    def extract_components(self):
        # reset each time
        self.labels = np.zeros_like(self.binary_image, dtype=np.uint32)
        self.current_label = 0
        component_sizes = {}
        
        for i in range(self.h):
            for j in range(self.w):
                if self.binary_image[i, j] == 1 and self.labels[i, j] == 0:
                    size = self.bfs(i, j)
                    component_sizes[self.current_label] = size
        return self.labels, component_sizes

    def largest_component_mask(self):
        labels, sizes = self.extract_components()
        if not sizes:
            return np.zeros_like(self.binary_image)
        largest_label = max(sizes, key=sizes.get)
        return (labels == largest_label).astype(np.uint8)


if __name__ == "__main__":
    cc = ConnectedComponents("ip_images/quote.png")
    labels, sizes = cc.extract_components()
    largest_mask = cc.largest_component_mask()

    print("Component sizes:", sizes)

    plt.plot(); plt.imshow(largest_mask, cmap="gray"); plt.title("Largest")
    plt.show()


