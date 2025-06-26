import os

import numpy as np
import pyvista as pv
import torch

import utils


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def line_points(points: np.ndarray):
    line = polyline_from_points(points)
    line["scalars"] = np.arange(line.n_points)
    tube = line.tube(radius=2)
    return tube


def pv_series(p, points):
    p.add_mesh(line_points(points[:, 0, :]), point_size=20.0, render_points_as_spheres=True)
    p.add_mesh(line_points(points[:, 1, :]), point_size=20.0, render_points_as_spheres=True)
    p.add_mesh(line_points(points[:, 2, :]), point_size=20.0, render_points_as_spheres=True)
    p.add_mesh(line_points(2 * points[:, 0, :] - points[:, 1, :]), point_size=20.0, render_points_as_spheres=True)
    p.add_mesh(line_points(2 * points[:, 0, :] - points[:, 2, :]), point_size=20.0, render_points_as_spheres=True)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    dir_save = r'../save/online_fm-hp_fm-Spine/RecON'

    n = len(os.listdir(dir_save)) // 2
    for idx in range(0, n):
        source = torch.load(os.path.join(dir_save, 'source_' + str(idx) + '.pth'), map_location=device)
        value = torch.load(os.path.join(dir_save, 'value_' + str(idx) + '.pth'), map_location=device)

        volume_real = utils.reconstruction.reco(source[0].squeeze(0).squeeze(1), value['real_series'].detach())
        volume = utils.reconstruction.reco(source[0].squeeze(0).squeeze(1), value['fake_series'][-1].detach())

        pv.set_plot_theme('document')
        p = pv.Plotter(shape=(1, 2))
        p.subplot(0, 0)
        p.add_text('GT', position='upper_left')
        p.add_volume(volume_real[0].cpu().numpy(), cmap='gray', mapper='gpu')
        pv_series(p, value['real_series'].detach().cpu().numpy() - volume_real[1].cpu().numpy())
        p.show_bounds(location='origin')
        p.subplot(0, 1)
        p.add_text('RecON', position='upper_left')
        p.add_volume(volume[0].cpu().numpy(), cmap='gray', mapper='gpu')
        pv_series(p, value['fake_series'][-1].detach().cpu().numpy() - volume[1].cpu().numpy())
        p.show_bounds(location='origin')
        p.link_views()
        p.add_axes()
        p.show()
