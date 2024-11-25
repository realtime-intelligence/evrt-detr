import numpy as np

def unsqueeze_frame(frame):
    # frame : (2*T, H, W)
    t, r = divmod(frame.shape[0], 2)

    assert r == 0
    # frame : (2, T, H, W)
    frame = frame.reshape(2, t, frame.shape[1], frame.shape[2])

    return frame

def accumulate_frame(frame):
    # frame : (2, T, H, W) -> (2, H, W)
    return frame.sum(axis = 1)

def frame_to_rgb_array(frame, symlog = 10, background = 0.7):
    # frame  : (2, H, W)
    # result : (H, W, 3)

    frame_pos = frame[0]
    frame_neg = frame[1]

    if symlog:
        sym_mask_pos = frame_pos > symlog
        sym_mask_neg = frame_neg > symlog
        frame_pos[sym_mask_pos] = (
            symlog + np.log(frame_pos[sym_mask_pos] / symlog)
        )
        frame_neg[sym_mask_neg] = (
            symlog + np.log(frame_neg[sym_mask_neg] / symlog)
        )

    result = np.full((*frame_pos.shape, 3), np.nan, dtype = np.float32)

    result[frame_pos > 0, 0] = frame_pos[frame_pos > 0]
    result[frame_neg > 0, 2] = frame_neg[frame_neg > 0]

    norm   = np.nanmax(result)
    result = result / norm

    result[np.all(np.isnan(result), axis = 2), :] = background
    result[np.isnan(result)] = 0

    return result

