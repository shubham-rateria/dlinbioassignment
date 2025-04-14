import math

def conv3d_output_dim(input_dims, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output dimensions for a 3D convolution layer.
    
    Args:
        input_dims (tuple): The input dimensions as (depth, height, width).
        kernel_size (int or tuple): The size of the kernel. Can be an int or a tuple (kD, kH, kW).
        stride (int or tuple): The stride of the convolution. Default is 1. Can be an int or tuple.
        padding (int or tuple): The padding applied on all sides. Default is 0. Can be an int or tuple.
        dilation (int or tuple): The dilation rate. Default is 1. Can be an int or tuple.
    
    Returns:
        tuple: Output dimensions as (depth_out, height_out, width_out).
    """
    # Ensure parameters are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    d_in, h_in, w_in = input_dims
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    d_out = math.floor((d_in + 2 * pD - dD * (kD - 1) - 1) / sD + 1)
    h_out = math.floor((h_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
    w_out = math.floor((w_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1)
    
    return d_out, h_out, w_out

# Example usage:
if __name__ == "__main__":
    # Define input dimensions: (depth, height, width)
    input_dims = (24, 128, 128)
    kernel_size = 4     # or (3, 3, 3)
    stride = 2           # or (2, 2, 2)
    padding = 1          # or (1, 1, 1)
    dilation = 1         # or (1, 1, 1)
    
    output_dims = conv3d_output_dim(input_dims, kernel_size, stride, padding, dilation)
    print("Output dimensions for the 3D convolution layer:", output_dims)
