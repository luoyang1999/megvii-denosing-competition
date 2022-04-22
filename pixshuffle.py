def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
        where s refers to scale factor
    """
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of inverted pixel unshuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C//(s*s), H*s, W*s],
        where s refers to scale factor
    """

    num, ch, height, width = tensor.shape

    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('chanel of tensor must be divisible by scale_factor.')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape([num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor
