from matplotlib import animation
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

class PauseAnimation:
    """Creates a plt animation with support to Pause when a key is pressed.

    If the key is registered by default, does not pause."""

    # based on keymaps declared in matplotlibrc
    DEFAULT_KEY_BINDINGS = "fhrcvposqgGlkL"

    def __init__(self, rec_name, image, emb, interval=200, start=0, stop=None,):
        self.rec_name = rec_name
        self.image = image
        self.emb = emb
        self.trace = emb.trace
        self.interval = interval
        self.start = start
        if stop != None:
            self.stop = stop
            self.image = image[start:stop]
        else:
            self.stop = len(image)-1
            self.image = image[start:-1]
        self.paused = False
        self.fig, self.ax = self.initialize_canvas()


    def initialize_canvas(self):
        fig = plt.figure(figsize=(14, 8), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[0.25, 0.75], hspace=0.25, wspace=0.05)

        ax = {}
        ax["top"] = fig.add_subplot(gs[0, :])    # top, full width
        ax["btm"] = fig.add_subplot(gs[1, :])    # bottom, full width

        # activity
        trace = self.trace
        dff = trace.dff
        dff[self.stop:] = np.nan
        self.dff = dff[self.start:]
        time = trace.time
        time = (trace.time/60)-30
        self.time = time[:len(self.dff)]

        self.dff_plot = ax["top"].plot(self.time, self.dff, color="black")
        self.frame_line = ax["top"].axvline(0, color="red", linewidth=3)
        lp_idxs, _ = trace.find_localpeaks()
        p_idxs = trace.peak_idxes 
        self.peaks = [ax["top"].axvline(self.time[p_idx- self.start], color='black', linewidth=3, alpha=0.35) for p_idx in p_idxs]
        self.local_peaks = [ax["top"].axvline(self.time[lp_idx- self.start], color='blue', linewidth=3, alpha=0.25) for lp_idx in lp_idxs]
        ax["top"].set_title(self.rec_name)
        ax["top"].set_xlabel("Time (min)", fontsize=15)
        xticks = np.arange(0, 360 + 60, 60, int)
        ax["top"].set_xticks(xticks)
        ax["top"].set_ylabel("DFF", fontsize=15)
        ax["top"].set_yticks([0, 1])
        ax["top"].spines['top'].set_visible(False)
        ax["top"].spines['right'].set_visible(False)
        ax["top"].spines['left'].set_linewidth(2.0)
        ax["top"].spines['bottom'].set_linewidth(2.0)
        ax["top"].tick_params(axis='both', which='major', width=2.0, length=6, labelsize=15)

        # movie
        ax["btm"].set_axis_off()
        initial_frame = self.image[0]
        # vmin, vmax = np.percentile(initial_frame, [0, 100])
        vmin = 25
        vmax = 600
        print(vmin, vmax)
        self.img_plot = ax["btm"].imshow(initial_frame, vmin=vmin, vmax=vmax)

        # frame text
        self.frame_num = ax["btm"].text(0.01, 0.98, f"f: {str(self.start)}\ninter: {str(self.interval)} ms", color="w", ha="left", va="top", transform=ax["btm"].transAxes)
        self.movie = ax["btm"].text(0.98, 0.02, f"{self.rec_name}", fontsize=8, color="w", ha="right", va="bottom", transform=ax["btm"].transAxes)

        fig.canvas.mpl_connect("key_press_event", self.toggle_pause)
        self.ani = animation.FuncAnimation(
            fig=fig,
            func=self.update,
            frames=len(self.image) - 1,
            interval=self.interval, # delay between frames in miliseconds
            repeat=True,
        )
        return fig, ax

    def update(self, frame):
        this_frame = frame
        self.img_plot.set_data(self.image[frame])
        self.frame_line.set_xdata([self.time[this_frame], self.time[this_frame]])
        self.frame_num.set_text(f"f: {str(this_frame)}\ninter: {str(self.interval)} ms")

        return self.img_plot

    def toggle_pause(self, event):
        if event.key in self.DEFAULT_KEY_BINDINGS:
            return
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused

    def display(self):
        plt.show()

    def save(self, filename):
        print(f"Saving animation as {filename}.gif...")
        self.ani.save(f"{filename}.gif", writer="pillow", fps=500)


class ContourAnimation(PauseAnimation):
    """Overlays ROI contour on top of a movie."""

    def __init__(self, image, contours, step_size, interval=50):
        super().__init__(image, interval)

        self.contours = contours
        self.paint_axes()
        self.step_size = step_size

    def paint_axes(self):
        x = self.contours[0][:, 0]
        y = self.contours[0][:, 1]
        self.contour_plot = self.ax.plot(y, x, color="red")[0]

    def update(self, frame):
        self.img_plot.set_data(self.image[frame])
        if frame % self.step_size == 0:
            i = frame // self.step_size
            x, y = self.contours[i][:, 0], self.contours[i][:, 1]
            self.contour_plot.set_data(y, x)
            self.frame_num.set_text(str(frame))

        return self.img_plot
