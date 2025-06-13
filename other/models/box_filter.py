import torch
from torch import nn


# cumulative sums method realisation
def diff_x(input, r):
    """
        Compute differences along the width dimension for the cumulative sums method.

        Used in the box filter to calculate the sum over a window of size (2r+1) by
        subtracting cumulative sums at appropriate offsets.

        Args:
            input (torch.Tensor): 4D input tensor, shape (N, C, H, W), typically a cumulative sum.
            r (int): Radius of the box filter window (window size is 2r+1).

        Returns:
            torch.Tensor: 4D tensor with differences, shape (N, C, H, W).

        Notes:
            Assumes the input tensor has padding of size r to avoid boundary issues.
        """
    assert input.dim() == 4,f"Expected 4D tensor, got {input.dim()}D"
    # Compute differences for left, middle, and right regions
    left = input[:, :, r:2 * r + 1] # Left boundary (first r rows)
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1] # Middle differences
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:-r - 1] # Right boundary
    # Concatenate along width dimension
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(input, r):
    """
        Compute differences along the height dimension for the cumulative sums method.

        Used in the box filter to calculate the sum over a window of size (2r+1) by
        subtracting cumulative sums at appropriate offsets.

        Args:
            input (torch.Tensor): 4D input tensor, shape (N, C, H, W), typically a cumulative sum.
            r (int): Radius of the box filter window (window size is 2r+1).

        Returns:
            torch.Tensor: 4D tensor with differences, shape (N, C, H, W).

        Notes:
            Assumes the input tensor has padding of size r to avoid boundary issues.
        """
    assert input.dim() == 4 , f"Expected 4D tensor, got {input.dim()}D"
    # Compute differences for top, middle, and bottom regions
    left = input[:, :, :, r:2 * r + 1] # Top boundary (first r columns)
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1] # Middle differences
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1] # Bottom boundary
    # Concatenate along height dimension
    output = torch.cat([left, middle, right], dim=3)
    return output


class BoxFilter(nn.Module):
    """
        PyTorch module for applying a box filter using the cumulative sums method.

        Computes the sum over a (2r+1) x (2r+1) window for each pixel using an integral
        image approach, which is efficient for large filter sizes.
        """
    def __init__(self, r):
        """
                Args:
                    r (int): Radius of the box filter (window size is 2r+1).
                """
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        """
                Apply the box filter to the input tensor.

                Args:
                    x (torch.Tensor): 4D input tensor, shape (N, C, H, W).

                Returns:
                    torch.Tensor: 4D output tensor, shape (N, C, H, W), containing the sum
                                  over the (2r+1) x (2r+1) window for each pixel.

                Notes:
                    - Assumes input has padding of size r to handle boundaries.
                    - To compute the mean (standard box filter), divide the output by (2r+1)^2.
                    - Example: output = box_filter(x) / ((2 * r + 1) ** 2)
                """
        assert x.dim() == 4 , f"Expected 4D tensor, got {x.dim()}D"
        # Compute cumulative sums along width and height
        # Apply differences along width and height to get window sums
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)
