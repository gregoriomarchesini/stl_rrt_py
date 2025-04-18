import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import json

class MapEditor2D:
    def __init__(self, bounds=(0, 10, 0, 10),filename :str ="map2d.json"):
        self.bounds = bounds
        self.rectangles = []
        self.start_point = None
        self.current_type = 'obstacle'

        # Setup figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("2D Map Editor")
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_aspect('equal')

        # UI elements
        plt.subplots_adjust(bottom=0.2)

        # Type selection buttons
        ax_obs = plt.axes([0.05, 0.1, 0.1, 0.05])
        ax_int = plt.axes([0.16, 0.1, 0.1, 0.05])
        self.btn_obs = Button(ax_obs, 'Obstacle')
        self.btn_int = Button(ax_int, 'Interest')

        self.btn_obs.on_clicked(lambda e: self.set_type('obstacle'))
        self.btn_int.on_clicked(lambda e: self.set_type('interest'))

        # Name input text box
        ax_name = plt.axes([0.3, 0.1, 0.4, 0.05])
        self.text_name = TextBox(ax_name, 'Name:', initial='')
        self.text_name.on_submit(self.update_object_name)
        self.text_name.set_active(False)

        # Save button
        ax_save = plt.axes([0.7, 0.1, 0.2, 0.05])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(lambda e: self.save_to_json(filename))

        self.temp_rect = None
        self.pressed = False
        self.last_created_index = None

        # Connect event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.selected_patch = None
        self.selected_data = None
        self.drag_offset = None

    def set_type(self, t):
        self.current_type = t
        self.ax.set_title(f"2D Editor - Mode: {self.current_type}")
        self.fig.canvas.draw()

    def update_object_name(self, text):
        if not self.rectangles or self.last_created_index is None:
            return
        self.rectangles[self.last_created_index]['name'] = text
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        # Try selecting an existing rectangle
        for i, patch in enumerate(self.ax.patches):
            contains, _ = patch.contains(event)
            if contains:
                self.selected_patch = patch
                self.selected_data = self.rectangles[i]
                self.last_created_index = i
                self.drag_offset = (
                    event.xdata - self.selected_data['x'],
                    event.ydata - self.selected_data['y']
                )
                self.text_name.set_val(self.selected_data.get('name', ''))
                self.text_name.set_active(True)
                return

        # Otherwise, start drawing a new one
        self.pressed = True
        self.start_point = (event.xdata, event.ydata)
        self.temp_rect = plt.Rectangle((event.xdata, event.ydata), 0, 0,
                                    edgecolor='r' if self.current_type == 'obstacle' else 'g',
                                    facecolor='none', lw=2)
        self.ax.add_patch(self.temp_rect)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.selected_patch and self.drag_offset:
            # Drag selected rectangle
            dx, dy = self.drag_offset
            new_x = event.xdata - dx
            new_y = event.ydata - dy

            self.selected_patch.set_xy((new_x, new_y))
            self.selected_data['x'] = new_x
            self.selected_data['y'] = new_y
            self.fig.canvas.draw_idle()
            return

        if not self.pressed or self.start_point is None:
            return

        # Continue drawing
        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        self.temp_rect.set_xy((x, y))
        self.temp_rect.set_width(w)
        self.temp_rect.set_height(h)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.selected_patch:
            # Reset the selection after dragging
            self.selected_patch = None
            self.selected_data = None
            self.drag_offset = None
            self.fig.canvas.draw_idle()
            return

        if not self.pressed or self.start_point is None:
            return

        self.pressed = False
        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        self.temp_rect.set_xy((x, y))
        self.temp_rect.set_width(w)
        self.temp_rect.set_height(h)

        box = {
            "type": self.current_type,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "name": f"{self.current_type}_{len(self.rectangles)+1}"
        }
        self.rectangles.append(box)
        self.last_created_index = len(self.rectangles) - 1
        self.text_name.set_val(box['name'])
        self.text_name.set_active(True)

        self.start_point = None
        self.temp_rect = None
        self.fig.canvas.draw_idle()

    def save_to_json(self, filename):
        """Save objects in center/size format to JSON file"""
        objects_list = []
        for obj in self.rectangles:
            center_based_obj = {
                "name": obj.get("name", ""),
                "type": obj["type"],
                "center_x": obj["x"] + obj["width"]/2,
                "center_y": obj["y"] + obj["height"]/2,
                "size_x": obj["width"],
                "size_y": obj["height"]
            }
            objects_list.append(center_based_obj)
        
        with open(filename, 'w') as f:
            json.dump(objects_list, f, indent=2)
        
        print(f"Saved {len(objects_list)} objects to {filename}")

    def run(self):
        plt.show()
        return self.rectangles



if __name__ == "__main__":
    
    editor = MapEditor2D(bounds=(-10,10,-10,10))
    boxes = editor.run()