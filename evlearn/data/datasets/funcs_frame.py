import torch

def apply_video_transforms(image, labels, transform_video, video_seed):
    if transform_video is None:
        return (image, labels)

    global_rng_state = torch.get_rng_state()
    torch.manual_seed(video_seed)

    if labels is not None:
        image, labels = transform_video(image, labels)
    else:
        image = transform_video(image)

    torch.set_rng_state(global_rng_state)

    return (image, labels)

def apply_transforms_to_frame(
    image, labels, transform_video, transform_frame, transform_labels,
    dtype, squash_time_polarity, video_seed
):
    # pylint: disable=too-many-arguments

    # image : (P, T, H, W)
    image = image.astype(dtype)
    image = torch.from_numpy(image)

    if squash_time_polarity:
        image = image.reshape((-1, *image.shape[-2:]))

    if transform_video is not None:
        image, labels = apply_video_transforms(
            image, labels, transform_video, video_seed
        )

    if transform_frame is not None:
        if labels is not None:
            image, labels = transform_frame(image, labels)
        else:
            image = transform_frame(image)

    if (labels is not None) and (transform_labels is not None):
        labels = transform_labels(labels)

    return (image, labels)

def apply_transforms_to_data(
    data, labels, transform_video, transform_frame, transform_labels,
    squash_time_polarity, video_seed
):
    # pylint: disable=too-many-arguments
    if len(data) > 0:
        frame = data[0]

    if squash_time_polarity:
        # frame : (P, T, H, W)
        #      -> (P*T,  H, W)
        frame = frame.reshape((-1, *frame.shape[-2:]))

    if transform_video is not None:
        frame, labels = apply_video_transforms(
            frame, labels, transform_video, video_seed
        )

    if transform_frame is not None:
        if labels is not None:
            frame, labels = transform_frame(frame, labels)
        else:
            frame = transform_frame(frame)

    if (labels is not None) and (transform_labels is not None):
        labels = transform_labels(labels)

    data = [ frame, *data[1:] ]

    return (data, labels)

