from matplotlib import animation
import matplotlib.pyplot as plt


class PauseAnimation:
    """Creates a plt animation with support to Pause when any key is pressed."""

    def __init__(self, image, interval=200):
        self.image = image
        self.interval = interval
        self.paused = False
        self.fig, self.ax = self.initialize_canvas()

    def initialize_canvas(self):
        fig, ax = plt.subplots()
        ax.set_axis_off()

        initial_frame = self.image[0]
        self.img_plot = ax.imshow(initial_frame)

        self.ani = animation.FuncAnimation(
            fig=fig,
            func=self.update,
            frames=len(self.image) - 1,
            interval=self.interval,
            repeat=True,
        )

        self.frame_num = ax.text(0, 5, str(0), color="w")
        fig.canvas.mpl_connect("key_press_event", self.toggle_pause)
        return fig, ax

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
        self.frame_num.set_text(str(frame))
        return self.img_plot

    def toggle_pause(self, *args):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def display(self):
        plt.show()

    def save(self, filename):
        print(f"Saving animation as {filename}.mp4...")
        self.ani.save(f"{filename}.mp4")


class ContourAnimation(PauseAnimation):
    """Overlays ROI contour on top of a movie."""

    def __init__(self, image, contours, interval=50):
        super().__init__(image, interval)
        self.contours = contours
        self.paint_axes()
        self.offset = image.shape[0] // len(contours)

    def paint_axes(self):
        x = self.contours[0][:, 0]
        y = self.contours[0][:, 1]
        self.contour_plot = self.ax.plot(y, x, color="red")[0]

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
        if frame % self.offset == 0:
            i = frame // self.offset
            x, y = self.contours[i][:, 0], self.contours[i][:, 1]
            self.contour_plot.set_data(y, x)
            self.frame_num.set_text(str(frame))
        return self.img_plot
