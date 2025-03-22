import numpy as np
import torch
import torch.nn.functional as F
from kornia.contrib import distance_transform
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree


class ImageProcessor:
    # images (torch.Tensor): (B, 1, H, W)
    @staticmethod
    def find_boundary_pixels(batch_tensor):
        device = batch_tensor.device
        batch_size, channel_size, height, width = batch_tensor.shape
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        padded = F.pad(batch_tensor.float(), (1, 1, 1, 1), mode='replicate')
        eroded = F.conv2d(padded, kernel, stride=1, groups=channel_size) == kernel.sum()
        closed = F.conv2d(padded, kernel, stride=1, groups=channel_size) > 0
        boundaries = (closed & ~eroded).float()
        return boundaries  # (B, C, H, W)

    @staticmethod
    def binarize_array(tensor):
        batch_size, channel_size, height, width = tensor.shape
        device = tensor.device
        # Scale to [0, 255] and round
        scaled = (tensor * 255).round()
        scaled = scaled.to(torch.float32)
        # Compute histogram for each channel in batch
        hist = torch.histc(scaled, bins=256, min=0, max=255, out=None).reshape(batch_size, channel_size, 256)
        # Compute cumulative sums
        bins = torch.arange(256, device=device, dtype=torch.float32).view(1, 1, -1)
        wB = torch.cumsum(hist, dim=-1)
        wF = wB[:, :, -1:] - wB
        valid = (wB > 0) & (wF > 0)
        # Compute means
        mB = torch.cumsum(hist * bins, dim=-1)
        mF = (mB[:, :, -1:] - mB) / wF.clamp(min=1e-6)
        mB = mB / wB.clamp(min=1e-6)
        # Compute inter-class variance
        between = valid * wB * wF * (mB - mF) ** 2
        # Find threshold index
        _, optimal_index = torch.max(between, dim=-1)
        # Normalize threshold to [0,1]
        thresholds = optimal_index.float() / 255.0
        thresholds = thresholds.view(batch_size, channel_size, 1, 1)
        # Apply threshold
        binary = (tensor > thresholds).float()  # Исправлено
        return binary, thresholds


class DistanceCalculator:
    # output_array = (b,class_count=1,h,w); targets = (b,h,w)
    def __init__(self, target_array, output_array=None, binarizer=ImageProcessor):
        self.target_array = target_array
        # target_array_binary.shape -> (B,C,H,W)
        self.target_array_binary, _ = binarizer.binarize_array(self.target_array.unsqueeze(1).detach())
        self.target_binary_boundary_array = binarizer.find_boundary_pixels(self.target_array_binary)
        self.device = None
        # output_array_binary.shape -> (B,C,H,W)
        if output_array:
            self.device = output_array.device
            self.output_array = output_array
            self.output_array_binary, _ = binarizer.binarize_array(output_array.detach().clone())

    def compute_distance_matrix_with_Kornia(self):
        distance_map = distance_transform(self.target_binary_boundary_array)
        return distance_map

    # for static data mask precomputing
    # Для boundary_loss (b,c,h,w) будет передоваться и браться where delta(S) область (b,c,h,w)
    def compute_distance_matrix_with_distance_transform_edt(self):
        # (GPU → CPU → GPU).
        batch_size, channel_size, height, width = self.target_binary_boundary_array.shape
        # Маска останется бинарной (0 и 1, а не 0 и 255).
        target_binary_boundary_array = self.target_binary_boundary_array.cpu().numpy().astype(np.uint8)

        inverted_image = 1 - target_binary_boundary_array
        # np.stack([...]) собирает результат в один массив размерности (B*C, H, W).
        # Функция distance_transform_edt из модуля scipy.ndimage вычисляет для каждого элемента, отличного от нуля (foreground)
        # , расстояние до ближайшего нулевого элемента (background) и заменяет значение этого элемента на вычисленное расстояние. Таким образом,
        # для всех элементов, равных 1, функция определяет расстояние до ближайшего элемента, равного 0, и записывает это расстояние на место исходной единицы.
        distance_maps = np.stack([
            distance_transform_edt((inverted_image[b, c]))
            for b in range(batch_size)
            for c in range(channel_size)
        ]).reshape(batch_size, channel_size, height, width)
        return torch.from_numpy(distance_maps).float().to(self.device) if self.device else torch.from_numpy(
            distance_maps).float()



    # Menq kd_trees-i hamar chenq pahum mask ayl petqa paheinq KDTree car@  boundary_points-ov stacvogh
    # for synthetic data mask computing
    # реализуем пока с помощью kdtree
    @staticmethod
    def _compute_distances_with_kd_trees(boundary_points, points):
        tree = KDTree(boundary_points)
        min_distances = np.zeros(len(points))
        second_min_distances = np.zeros(len(points))

        for idx, point in enumerate(points):
            nearest_distances, _ = tree.query(point, k=2)
            min_distances[idx] = nearest_distances[:, 0]
            second_min_distances[idx] = nearest_distances[:, 1]

        return min_distances, second_min_distances


    # stegh sagh pixel-neri hamara hashvum erkar process-a, boudnary-ium aveli meghma menak delta(S)-i hamar
    # full cpu usage
    def compute_min_and_second_min_distance_with_kdtree(self):
        # target_array_binary.shape -> (B,C,H,W)
        batch_size, channel_size, height, width = self.target_binary_boundary_array.shape
        target_binary_boundary_array = self.target_binary_boundary_array
        # min_distances = self.compute_distance_matrix_with_Kornia()

        min_distances = self.compute_distance_matrix_with_distance_transform_edt()
        second_min_distances = np.zeros((batch_size, channel_size, height, width))

        for b in range(batch_size):
            for c in range(channel_size):
                boundary_points = np.argwhere(target_binary_boundary_array[b, c, :, :] == 1)
                empty_points = np.argwhere(target_binary_boundary_array[b, c, :, :] == 0)

                if boundary_points.size == 0:
                    raise ValueError('No boundary points found')

                min_dist, sec_min_dist = self._compute_distances_with_kd_trees(boundary_points, empty_points)

                for idx, point in enumerate(empty_points):
                    i, j = point
                    min_distances[b, c, i, j] = min_dist[idx]
                    second_min_distances[b, c, i, j] = sec_min_dist[idx]
        return torch.from_numpy(min_distances).float().to(self.device), torch.from_numpy(
            second_min_distances).float().to(self.device)
