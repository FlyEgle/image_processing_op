import numpy as np
import matplotlib.pyplot as plt
import cv2


# padding operate
def padding_op(x, is_padding, mode, padding):
    """
    inputs:
          x : input_array
          is_padding: True return array with padding mode and padding value
          mode: "SAME" -> padding = [value, value, value, value]
                "ROW"  -> padding = [value, 0, 0, value]
                "COLUMNS" -> padding = [0, value, value, 0]
                "SINGLE_ZEROS" -> padding = [0, value, value, value]
                "SINGLE_PADDING" -> padding = [0, 0, 0, value]
    outputs:
          y: output_array
    """
    # padding
    if is_padding:
        # up, left, down, right
        shape = x.shape
        y = np.zeros((shape[0] + padding[0] + padding[3], shape[1] + padding[1] + padding[2]))
        # top, left, right, down all padding zeros
        if mode == "SAME" or mode == "same":
            if not np.all(padding) >= 1:
                raise ValueError("mode and padding is not suitable, change both with them")
            y[padding[0]:-padding[3], padding[1]:-padding[2]] = x
            return y
        # top and down
        elif mode == "row" or mode == "ROW":
            if padding[0] and padding[3] == 0:
                raise ValueError("row mode need padding=[value, 0, 0, value]")
            y[padding[0]:-padding[3], :] = x
            return y
        # left and right
        elif mode == "columns" or mode == "COLUMNS":
            if padding[1] and padding[2] == 0:
                raise ValueError("columns mode need padding=[0, value, value, 0]")
            y[:, padding[1]:-padding[2]] = x
            return y
        # single zeros padding
        elif mode == "SINGLE_ZEROS" or mode == "single_zeros":
            # top
            if padding[0] == 0:
                y[:-padding[3], padding[1]:-padding[2]] = x
                return y
            # down
            elif padding[3] == 0:
                y[padding[1]:, padding[1]:-padding[2]] = x
                return y
            # left
            elif padding[1] == 0:
                y[padding[0]:-padding[3], :-padding[2]] = x
                return y
            # right
            elif padding[2] == 0:
                y[padding[0]:-padding[3], padding[1]:] = x
                return y
        # single value padding
        else:
            if padding[0] > 0:
                y[padding[0]:, :] = x
                return y
            elif padding[1] > 0:
                y[:, padding[1]:] = x
                return y
            elif padding[2] > 0:
                y[:, :-padding[2]] = x
                return y
            else:
                y[:-padding[3], :] = x
                return y
    # no padding
    else:
        return x
