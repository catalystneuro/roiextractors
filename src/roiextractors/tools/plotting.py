def show_video(imaging, ax=None):
    """Show video as animation.

    Parameters
    ----------
    imaging: ImagingExtractor
        The imaging extractor object to be saved in the .h5 file
    ax: matplotlib axis
        Axis to plot the video. If None, a new axis is created.

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation of the video.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    def animate_func(i, imaging, im, ax):
        ax.set_title(f"{i}")
        im.set_array(imaging.get_frames([i])[0])
        return [im]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
    im0 = imaging.get_frames([0])[0]
    im = ax.imshow(im0, interpolation="none", aspect="auto", vmin=0, vmax=1)
    interval = 1 / imaging.get_sampling_frequency() * 1000
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=imaging.get_num_frames(),
        fargs=(imaging, im, ax),
        interval=interval,
        blit=False,
    )
    return anim
