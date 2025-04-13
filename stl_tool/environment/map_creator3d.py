import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

class MapEditor3d:
    def __init__(self, bounds=(0, 10, 0, 10), default_height=2.0, filename="map3d.json"):
        self.bounds = bounds
        self.default_height = default_height
        self.rectangles = []
        self.start_point = None
        self.current_type = 'obstacle'

        # Setup figure and axes
        self.fig = plt.figure(figsize=(12, 6))
        self.ax2d = self.fig.add_subplot(1, 2, 1)
        self.ax3d = self.fig.add_subplot(1, 2, 2, projection='3d')

        self.ax2d.set_title("2D Editor")
        self.ax2d.set_xlim(bounds[0], bounds[1])
        self.ax2d.set_ylim(bounds[2], bounds[3])
        self.ax2d.set_aspect('equal')

        self.ax3d.set_title("3D Preview")
        self.ax3d.set_xlim(bounds[0], bounds[1])
        self.ax3d.set_ylim(bounds[2], bounds[3])
        self.ax3d.set_zlim(0, 5)

        # UI elements
        plt.subplots_adjust(bottom=0.35)

        # Type selection buttons
        ax_obs = plt.axes([0.05, 0.25, 0.1, 0.05])
        ax_int = plt.axes([0.16, 0.25, 0.1, 0.05])
        self.btn_obs = Button(ax_obs, 'Obstacle')
        self.btn_int = Button(ax_int, 'Interest')

        self.btn_obs.on_clicked(lambda e: self.set_type('obstacle'))
        self.btn_int.on_clicked(lambda e: self.set_type('interest'))

        # Height slider
        ax_slider = plt.axes([0.05, 0.2, 0.4, 0.03])
        self.slider = Slider(ax_slider, 'Height', 0.1, 5.0, valinit=self.default_height)
        self.slider.on_changed(self.update_height)

        # Lower and upper bound sliders for last object
        ax_lower = plt.axes([0.3, 0.15, 0.6, 0.03])
        ax_upper = plt.axes([0.3, 0.1, 0.6, 0.03])
        
        self.slider_lower = Slider(ax_lower, 'Lower Bound', 0, 5, valinit=0, valstep=0.1)
        self.slider_upper = Slider(ax_upper, 'Upper Bound', 0, 5, valinit=self.default_height, valstep=0.1)
        
        self.slider_lower.on_changed(self.update_last_object_bounds)
        self.slider_upper.on_changed(self.update_last_object_bounds)
        
        # Name input text box
        ax_name = plt.axes([0.3, 0.05, 0.6, 0.04])
        self.text_name = TextBox(ax_name, 'Name:', initial='')
        self.text_name.on_submit(self.update_object_name)
        
        # Disable sliders and text box initially (no objects yet)
        self.slider_lower.set_active(False)
        self.slider_upper.set_active(False)
        self.text_name.set_active(False)

        self.temp_rect = None
        self.pressed = False
        self.start_point = None
        self.last_created_index = None  # Track index of last created object

        # Connect event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.selected_patch = None
        self.selected_data = None
        self.drag_offset = None

        # Add save button
        ax_save = plt.axes([0.05, 0.1, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(lambda e: self.save_to_json(filename))
    
    def save_to_json(self, filename):
        # Convert from corner-based to center-based format
        objects_list = []
        for obj in self.rectangles:
            center_based_obj = {
                "name": obj.get("name", ""),
                "type": obj["type"],
                "center_x": obj["x"] + obj["width"]/2,
                "center_y": obj["y"] + obj["depth"]/2,
                "center_z": obj["z"] + obj["height"]/2,
                "size_x": obj["width"],
                "size_y": obj["depth"],
                "size_z": obj["height"]
            }
            objects_list.append(center_based_obj)
        
        with open(filename, 'w') as f:
            json.dump(objects_list, f, indent=2)
        
        print(f"Saved {len(objects_list)} objects to {filename}")

    def set_type(self, t):
        self.current_type = t
        self.ax2d.set_title(f"2D Editor – Mode: {self.current_type}")
        self.fig.canvas.draw()

    def update_height(self, val):
        self.default_height = val
        # Also update the upper bound slider to match the new height
        if self.slider_upper.val < val:
            self.slider_upper.set_val(val)
        
    def update_last_object_bounds(self, val):
        if not self.rectangles or self.last_created_index is None:
            return
            
        obj = self.rectangles[self.last_created_index]
        
        # Ensure upper bound is always >= lower bound
        if self.slider_upper.val < self.slider_lower.val:
            if self.slider_upper is self.sender():
                self.slider_lower.set_val(self.slider_upper.val)
            else:
                self.slider_upper.set_val(self.slider_lower.val)
            return
            
        obj['z'] = self.slider_lower.val
        obj['height'] = self.slider_upper.val - self.slider_lower.val
        self.update_3d_view()

    def update_object_name(self, text):
        if not self.rectangles or self.last_created_index is None:
            return
        self.rectangles[self.last_created_index]['name'] = text
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax2d:
            return

        # Try selecting an existing rectangle
        for i, patch in enumerate(self.ax2d.patches):
            contains, _ = patch.contains(event)
            if contains:
                self.selected_patch = patch
                self.selected_data = self.rectangles[i]
                self.last_created_index = i  # Track the selected object
                self.drag_offset = (
                    event.xdata - self.selected_data['x'],
                    event.ydata - self.selected_data['y']
                )
                
                # Update sliders and name to show selected object's properties
                self.slider_lower.set_val(self.selected_data['z'])
                self.slider_upper.set_val(self.selected_data['z'] + self.selected_data['height'])
                self.text_name.set_val(self.selected_data.get('name', ''))
                
                # Enable controls
                self.slider_lower.set_active(True)
                self.slider_upper.set_active(True)
                self.text_name.set_active(True)
                
                return  # Stop here to start dragging instead of drawing

        # Otherwise, start drawing a new one
        self.pressed = True
        self.start_point = (event.xdata, event.ydata)
        self.temp_rect = plt.Rectangle((event.xdata, event.ydata), 0, 0,
                                    edgecolor='r' if self.current_type == 'obstacle' else 'g',
                                    facecolor='none', lw=2)
        self.ax2d.add_patch(self.temp_rect)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if event.inaxes != self.ax2d:
            return

        if self.selected_patch and self.drag_offset:
            # Drag selected rectangle
            dx, dy = self.drag_offset
            new_x = event.xdata - dx
            new_y = event.ydata - dy

            self.selected_patch.set_xy((new_x, new_y))
            self.selected_data['x'] = new_x
            self.selected_data['y'] = new_y
            self.update_3d_view()
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
            "z": 0,
            "width": w,
            "depth": h,
            "height": self.default_height,
            "name": f"{self.current_type}_{len(self.rectangles)+1}"  # Default name
        }
        self.rectangles.append(box)
        self.last_created_index = len(self.rectangles) - 1  # Track the new object

        # Enable sliders and text box, set their values for the new object
        self.slider_lower.set_active(True)
        self.slider_upper.set_active(True)
        self.text_name.set_active(True)
        self.slider_lower.set_val(0)
        self.slider_upper.set_val(self.default_height)
        self.text_name.set_val(box['name'])

        self.start_point = None
        self.temp_rect = None
        self.update_3d_view()

    def update_3d_view(self):
        self.ax3d.cla()
        self.ax3d.set_title("3D Preview")
        self.ax3d.set_xlim(self.bounds[0], self.bounds[1])
        self.ax3d.set_ylim(self.bounds[2], self.bounds[3])
        self.ax3d.set_zlim(0, 5)

        for box in self.rectangles:
            self.draw_box(self.ax3d, box)
            
            # Add name label if it exists
            name = box.get('name', '')
            if name:
                x = box['x'] + box['width']/2
                y = box['y'] + box['depth']/2
                z = box['z'] + box['height']
                self.ax3d.text(x, y, z, name, color='black', ha='center', va='bottom')
                
        self.fig.canvas.draw_idle()

    def draw_box(self, ax, box):
        x, y, z = box['x'], box['y'], box['z']
        dx, dy, dz = box['width'], box['depth'], box['height']
        color = 'r' if box['type'] == 'obstacle' else 'g'

        corners = [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
        faces = [
            [corners[i] for i in [0, 1, 2, 3]],  # bottom
            [corners[i] for i in [4, 5, 6, 7]],  # top
            [corners[i] for i in [0, 1, 5, 4]],  # front
            [corners[i] for i in [1, 2, 6, 5]],  # right
            [corners[i] for i in [2, 3, 7, 6]],  # back
            [corners[i] for i in [3, 0, 4, 7]],  # left
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.4, edgecolors='k'))

    def run(self):
        plt.show()
        return self.rectangles


if __name__ == "__main__":
    editor = MapEditor3d()
    boxes = editor.run()
