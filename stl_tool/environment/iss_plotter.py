import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

class ISSModel:
    REAL_ISS_LENGTH = 108  # meters (approximate real ISS length)
    script_dir = os.path.dirname(__file__)
    stl_path = os.path.join(script_dir, "..", "_assets", "International Space Station.stl")

    def __init__(self):
        self.mesh = mesh.Mesh.from_file(self.stl_path)
        self._scale_to_real_size()

    def _scale_to_real_size(self):
        mins = self.mesh.vectors.min(axis=(0, 1))
        maxs = self.mesh.vectors.max(axis=(0, 1))
        dims = maxs - mins
        model_length = np.max(dims)

        print(f"Model bounding box (m): {dims}")
        print(f"Model length: {model_length:.2f} m")

        scale_factor = self.REAL_ISS_LENGTH / model_length
        print(f"Scaling mesh by factor: {scale_factor:.3f}")

        self.mesh.vectors *= scale_factor

    def plot(self, elev=30, azim=45):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        collection = Poly3DCollection(self.mesh.vectors, alpha=0.7, edgecolor='k')
        ax.add_collection3d(collection)

        scale = self.mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        ax.set_box_aspect([1, 1, 1])

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        return fig, ax


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    stl_path = os.path.join(script_dir, "..", "_assets", "International Space Station.stl")

    iss = ISSModel()
    iss.plot()