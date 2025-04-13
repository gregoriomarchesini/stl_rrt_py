import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import json

class InteractiveMapEditor:
    def __init__(self, map_bounds):
        self.map_bounds = map_bounds
        self.rectangles = []
        self.current_type = "obstacle"
        self.start_point = None
        self.temp_rect = None

        # Setup plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        self.ax.set_xlim(map_bounds[0], map_bounds[1])
        self.ax.set_ylim(map_bounds[2], map_bounds[3])
        self.ax.set_title(f"Current type: {self.current_type}")

        # Buttons
        self.ax_obstacle = plt.axes([0.05, 0.1, 0.15, 0.075])
        self.ax_interest = plt.axes([0.25, 0.1, 0.15, 0.075])
        self.ax_undo = plt.axes([0.45, 0.1, 0.15, 0.075])
        self.ax_save = plt.axes([0.65, 0.1, 0.15, 0.075])

        self.btn_obstacle = Button(self.ax_obstacle, 'Obstacle')
        self.btn_interest = Button(self.ax_interest, 'Interest')
        self.btn_undo = Button(self.ax_undo, 'Undo')
        self.btn_save = Button(self.ax_save, 'Save')

        self.btn_obstacle.on_clicked(self.set_obstacle)
        self.btn_interest.on_clicked(self.set_interest)
        self.btn_undo.on_clicked(self.undo_last)
        self.btn_save.on_clicked(self.save_rectangles)

        # Event connections
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def set_obstacle(self, event):
        self.current_type = "obstacle"
        self.ax.set_title(f"Current type: {self.current_type}")
        self.fig.canvas.draw()

    def set_interest(self, event):
        self.current_type = "interest"
        self.ax.set_title(f"Current type: {self.current_type}")
        self.fig.canvas.draw()

    def undo_last(self, event):
        if self.rectangles:
            last_patch = self.rectangles.pop()
            last_patch['rect'].remove()
            self.fig.canvas.draw()

    def save_rectangles(self, event):
        data = [
            {key: val for key, val in r.items() if key != "rect"}
            for r in self.rectangles
        ]
        with open("map_rectangles.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Saved to map_rectangles.json")

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        self.temp_rect = Rectangle(self.start_point, 0, 0, edgecolor='blue', facecolor='none', linestyle='--')
        self.ax.add_patch(self.temp_rect)
        self.fig.canvas.draw()

    def on_motion(self, event):
        if self.start_point and self.temp_rect and event.inaxes == self.ax:
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata
            x, y = min(x0, x1), min(y0, y1)
            width, height = abs(x1 - x0), abs(y1 - y0)
            self.temp_rect.set_bounds(x, y, width, height)
            self.fig.canvas.draw()

    def on_release(self, event):
        if not self.start_point or not self.temp_rect or event.inaxes != self.ax:
            return

        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        x, y = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)

        color = 'red' if self.current_type == 'obstacle' else 'green'
        final_rect = Rectangle((x, y), width, height, edgecolor=color, facecolor='none', linewidth=2)
        self.ax.add_patch(final_rect)

        self.rectangles.append({
            "type": self.current_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rect": final_rect
        })

        self.start_point = None
        self.temp_rect.remove()
        self.temp_rect = None
        self.fig.canvas.draw()

    def run(self):
        plt.show()
        return [
            {key: val for key, val in r.items() if key != "rect"}
            for r in self.rectangles
        ]


# Example usage
bounds = (0, 10, 0, 10)
editor = InteractiveMapEditor(bounds)
rectangles = editor.run()

print("Final rectangles:")
for r in rectangles:
    print(r)
