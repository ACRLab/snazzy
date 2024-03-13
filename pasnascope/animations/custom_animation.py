import matplotlib.pyplot as plt
from matplotlib import animation


class PauseAnimation:
    '''Creates a plt animation with support to Pause when any key is pressed.'''

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
            fig=fig, func=self.update, frames=len(self.image), interval=self.interval, repeat=False
        )
        fig.canvas.mpl_connect('key_press_event', self.toggle_pause)
        return fig, ax

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
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
        self.ani.save(f'{filename}.mp4')


class CentroidAnimation(PauseAnimation):
    '''Overlays centroid positions on top of a movie.'''

    def __init__(self, image, centroids, interval=50):
        super().__init__(image, interval)
        self.centroids = centroids
        self.paint_axes()

    def paint_axes(self):
        y, x = self.centroids[0]
        self.centroid_plot = self.ax.plot(y, x, marker='o', color='red')[0]

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
        y, x = self.centroids[frame]
        self.centroid_plot.set_data(x, y)
        return self.img_plot


class ContourAnimation(PauseAnimation):
    '''Overlays ROI contour on top of a movie.'''

    def __init__(self, image, contours, interval=50):
        super().__init__(image, interval)
        self.contours = contours
        self.paint_axes()
        self.offset = image.shape[0]//len(contours)

    def paint_axes(self):
        x = self.contours[0][:, 0]
        y = self.contours[0][:, 1]
        self.contour_plot = self.ax.plot(y, x, color='red')[0]

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
        if frame % self.offset == 0:
            i = frame // self.offset
            x, y = self.contours[i][:, 0], self.contours[i][:, 1]
            self.contour_plot.set_data(y, x)
        return self.img_plot
