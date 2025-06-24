import torch
import torch.nn.functional as F


def get_axis(series, eps=1e-20):
    old_dtype = series.dtype
    series = series.type(torch.float64)

    p1p2 = series[:, 1:3, :] - series[:, 0:1, :]
    ax_x = p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_x = F.normalize(ax_x, p=2, dim=-1, eps=eps)
    ax_y = -p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_y = F.normalize(ax_y, p=2, dim=-1, eps=eps)
    ax_z = torch.cross(ax_x, ax_y, dim=-1)
    ax_z = F.normalize(ax_z, p=2, dim=-1, eps=eps)
    axis = torch.stack([ax_x, ax_y, ax_z], dim=1)

    axis = axis.type(old_dtype)
    return axis


def _get_weight(dist, iter=2, temperature=0.001, eps=1e-10):
    weight = torch.reciprocal(dist + eps)
    w_iter = weight * torch.softmax(weight / temperature, dim=0)
    for _ in range(iter - 1):
        w_iter = weight * torch.softmax(torch.abs(weight - w_iter) / temperature, dim=0) + w_iter
    weight = w_iter / torch.sum(w_iter, dim=0, keepdim=True)
    return weight


def _reco_block(slices, matrix, mesh):
    n = len(slices)
    rm_shape = mesh.shape[:3]

    loca = torch.einsum('Nij,XYZj->NXYZi', matrix, mesh)

    flag = (torch.sum(~(loca[..., 2] < -0.5), dim=0) == 0) | (torch.sum(~(loca[..., 2] > 0.5), dim=0) == 0)
    weight = _get_weight(torch.abs(loca[..., 2]))
    grid = loca[..., :2].view(n, 1, -1, 2)
    grid = torch.cat([grid[..., 0:1] / (slices.shape[-2] / 2), grid[..., 1:2] / (slices.shape[-1] / 2)], dim=-1)

    value = F.grid_sample(slices.unsqueeze(1), grid.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
    value = value.view(n, *rm_shape)

    flag3 = grid.view(n, *rm_shape, 2)
    flag3 = (torch.abs(flag3[..., 0]) > 1) | (torch.abs(flag3[..., 1]) > 1)
    weight[flag3] = 0

    weight[:, flag] = 0
    volume = torch.einsum('NXYZ,NXYZ->XYZ', weight, value)
    return volume


def _reco_split(slices, matrix, reco_mesh, chunk_size=None):
    chunk_size = chunk_size or (50, 50, 50)
    cza = []
    for cz in torch.split(reco_mesh, chunk_size[2], dim=2):
        cya = []
        for cy in torch.split(cz, chunk_size[1], dim=1):
            cxa = []
            for cx in torch.split(cy, chunk_size[0], dim=0):
                cx = _reco_block(slices, matrix, cx)
                cxa.append(cx)
            cxa = torch.cat(cxa, dim=0)
            cya.append(cxa)
        cya = torch.cat(cya, dim=1)
        cza.append(cya)
    cza = torch.cat(cza, dim=2)
    return cza


def _reco_stack(down_source, matrix, reco_mesh, chunk_size=None):
    reco = []
    for idx in range(len(down_source)):
        reco.append(_reco_split(down_source[idx].squeeze(1), matrix[idx, ...], reco_mesh, chunk_size=chunk_size))
    reco = torch.stack(reco, dim=0).unsqueeze(1)
    return reco


def get_reco_size(series, mat_scale=None):
    series = torch.cat([series, 2 * series[:, 0:1, :] - series[:, 1:2, :], 2 * series[:, 0:1, :] - series[:, 2:3, :]], dim=1)
    min_point = torch.min(series.view(-1, 3), dim=0)[0]
    max_point = torch.max(series.view(-1, 3), dim=0)[0]
    range_point = max_point - min_point + 1
    if mat_scale is not None:
        range_point[0] *= mat_scale[0, 0]
        range_point[1] *= mat_scale[1, 1]
        range_point[2] *= mat_scale[2, 2]
    reco_size = torch.ceil(range_point).long().tolist()
    bias = min_point - 0.5
    return reco_size, bias


def get_matrix(series, mat_scale=None):
    axis = get_axis(series).permute(0, 2, 1)
    axis = torch.cat([axis, series[:, 0:1, :].permute(0, 2, 1)], dim=-1)
    axis = F.pad(axis, [0, 0, 0, 1])
    axis[:, -1, -1] = 1

    if mat_scale is not None:
        mat_scale_inv = torch.inverse(mat_scale)
        mat_scale = mat_scale.unsqueeze(0).expand(len(axis), 4, 4)
        mat_scale_inv = mat_scale_inv.unsqueeze(0).expand(len(axis), 4, 4)
        axis = torch.bmm(mat_scale, torch.bmm(axis, mat_scale_inv))

    axis = torch.inverse(axis)
    return axis


def transform(points, height, width):
    axis = get_axis(points).permute(0, 2, 1)

    if not hasattr(transform, 'mesh') or height != transform.height or width != transform.width:
        range_x = torch.arange(-(height - 1) / 2, (height + 1) / 2, dtype=points.dtype, device=points.device)
        range_y = torch.arange(-(width - 1) / 2, (width + 1) / 2, dtype=points.dtype, device=points.device)
        mesh_x, mesh_y = torch.meshgrid(range_x, range_y, indexing='ij')
        mesh = torch.stack([mesh_y, -mesh_x, torch.zeros_like(mesh_x)], dim=-1)
        transform.mesh = mesh
        transform.height = height
        transform.width = width

    center = points[:, 0, :].unsqueeze(1).unsqueeze(1)

    local_mesh = torch.einsum('Nij,HWj->NHWi', axis, transform.mesh) + center
    return local_mesh


def reco(source, series, mat_scale=None, volume_size=None):
    if volume_size is not None:
        reco_size = volume_size
        if not hasattr(reco, 'bias'):
            reco.bias = -torch.tensor(volume_size, dtype=series.dtype, device=series.device) / 2
        bias = reco.bias
    else:
        reco_size, bias = get_reco_size(series, mat_scale)
    series = series - bias

    matrix = get_matrix(series, mat_scale)
    matrix = torch.stack([-matrix[:, 1], matrix[:, 0], matrix[:, 2]], dim=1)

    if not hasattr(reco, 'reco_mesh') or reco_size != reco.reco_size:
        reco_mesh = torch.meshgrid([torch.arange(0.5, length + 0.5, dtype=source.dtype, device=series.device) for length in reco_size], indexing='ij')
        reco_mesh = torch.stack(reco_mesh, dim=-1)
        reco_mesh = F.pad(reco_mesh, (0, 1))
        reco_mesh[..., -1] = 1
        reco.reco_mesh = reco_mesh
        reco.reco_size = reco_size

    volume = _reco_stack(source.unsqueeze(0).unsqueeze(2), matrix.unsqueeze(0), reco.reco_mesh, chunk_size=volume_size).squeeze(0).squeeze(0)
    return volume, bias


def get_slice(volume, series, shape):
    mesh = transform(series, *shape)
    if not hasattr(get_slice, 'volume_size') or volume.shape != get_slice.shape:
        get_slice.volume_size = torch.tensor(volume.shape, dtype=mesh.dtype, device=mesh.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        get_slice.shape = volume.shape
    mesh = mesh.unsqueeze(0) / get_slice.volume_size
    mesh = (mesh * 2 - 1).flip(-1)
    slices = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), mesh, mode='bilinear', padding_mode='border', align_corners=False)
    slices = slices.squeeze(1).unsqueeze(2)
    return slices
