import math

import numpy as np
import torch
import torch.nn.functional as F


def get_axis(series, eps: float = 1e-20):
    old_dtype = series.dtype
    series = series.type(torch.float64)

    p1p2 = series[:, 1:3, :] - series[:, 0:1, :]
    ax_x = p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_x = F.normalize(ax_x, p=2.0, dim=-1, eps=eps)
    ax_y = -p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_y = F.normalize(ax_y, p=2.0, dim=-1, eps=eps)
    ax_z = torch.cross(ax_x, ax_y, dim=-1)
    ax_z = F.normalize(ax_z, p=2.0, dim=-1, eps=eps)
    axis = torch.stack([ax_x, ax_y, ax_z], dim=1)

    axis = axis.type(old_dtype)
    return axis


def get_normal(points, eps: float = 1e-20):
    old_dtype = points.dtype
    points = points.type(torch.float64)

    p1p2 = points[1:, :] - points[0:1, :]
    p1p2 = F.normalize(p1p2, p=2.0, dim=-1, eps=eps)
    normal = torch.cross(p1p2[0, :], p1p2[1, :], dim=-1)

    normal = normal.type(old_dtype)
    return normal


def get_quaternion_matrix(normal, sin_2, cos_2):
    old_dtype = normal.dtype
    normal = normal.type(torch.float64)
    sin_2 = sin_2.type(torch.float64) if isinstance(sin_2, torch.Tensor) else sin_2
    cos_2 = cos_2.type(torch.float64) if isinstance(cos_2, torch.Tensor) else cos_2

    bcd = sin_2 * normal
    b2, c2, d2 = (2 * bcd ** 2).split(1, dim=-1)
    ab, ac, ad = (2 * cos_2 * bcd).split(1, dim=-1)

    bc = 2 * bcd[:, 0:1] * bcd[:, 1:2]
    bd = 2 * bcd[:, 0:1] * bcd[:, 2:3]
    cd = 2 * bcd[:, 1:2] * bcd[:, 2:3]

    matrix = torch.cat([1 - c2 - d2, bc - ad, ac + bd, bc + ad, 1 - b2 - d2, cd - ab, bd - ac, ab + cd, 1 - b2 - c2], dim=-1)
    matrix = matrix.view(matrix.shape[0], 3, 3)

    matrix = matrix.type(old_dtype)
    return matrix


def euler_from_matrix(matrix, eps: float = 1e-6):
    i, j, k = 0, 1, 2
    M = matrix[:, :3, :3]

    cy = torch.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])
    ax = torch.atan2(M[:, k, j], M[:, k, k])
    ay = torch.atan2(-M[:, k, i], cy)
    az = torch.atan2(M[:, j, i], M[:, i, i])
    flag = cy <= eps
    ax2 = torch.atan2(-M[:, j, k], M[:, j, j])
    ax[flag, ...] = ax2[flag, ...]
    az[flag, ...] = 0

    a = torch.stack([ax, ay, az], dim=-1)
    return a


def euler_matrix(angle):
    i, j, k = 0, 1, 2
    ai, aj, ak = angle[:, 0], angle[:, 1], angle[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = torch.eye(4, dtype=ai.dtype, device=ai.device).unsqueeze(0).repeat(len(ai), 1, 1)
    M[:, i, i] = cj * ck
    M[:, i, j] = sj * sc - cs
    M[:, i, k] = sj * cc + ss
    M[:, j, i] = cj * sk
    M[:, j, j] = sj * ss + cc
    M[:, j, k] = sj * cs - sc
    M[:, k, i] = -sj
    M[:, k, j] = cj * si
    M[:, k, k] = cj * ci

    return M


def affine_matrix_from_points(v0, v1):
    t0 = -torch.mean(v0, dim=-1)
    v0 = v0 + t0.unsqueeze(-1)
    t1 = -torch.mean(v1, dim=-1)
    v1 = v1 + t1.unsqueeze(-1)

    u, s, vh = torch.svd(torch.bmm(v1, v0.permute(0, 2, 1)).cpu())
    if u.device != v0.device:
        u, vh = torch.cat([u, vh], dim=-1).to(v0.device).split(3, dim=-1)
    vh = vh.permute(0, 2, 1)
    R = torch.bmm(u, vh)

    flag = torch.det(R) < 0.0
    out = u[:, :, 2:3] * (vh[:, 2:3, :] * 2.0)
    R[flag, ...] = R[flag, ...] - out[flag, ...]

    M = torch.cat([R, torch.sum(R * t0.unsqueeze(1), dim=-1, keepdim=True) - t1.unsqueeze(-1)], dim=-1)
    M = F.pad(M, [0, 0, 0, 1])
    M[:, -1, -1] = 1.0
    return M


def quaternion_rotation_mul_theta(point, origin, normal, sin_2, cos_2):
    old_dtype = point.dtype
    point = point.type(torch.float64)
    origin = origin.type(torch.float64)
    normal = normal.type(torch.float64)
    sin_2 = sin_2.type(torch.float64) if isinstance(sin_2, torch.Tensor) else sin_2
    cos_2 = cos_2.type(torch.float64) if isinstance(cos_2, torch.Tensor) else cos_2

    point = point - origin
    matrix = get_quaternion_matrix(normal, sin_2, cos_2)
    next_points = (matrix @ point.T.unsqueeze(0).expand(matrix.shape[0], 3, 3)).permute(0, 2, 1)
    next_points = next_points + origin.unsqueeze(0)

    next_points = next_points.type(old_dtype)
    return next_points


def series_to_dof(series):
    old_dtype = series.dtype
    series = series.type(torch.float64)

    angle_mat = get_axis(series[:-1]).permute(0, 2, 1)
    angle_mat_inv = torch.inverse(angle_mat)

    p0p1 = torch.bmm(torch.cat([angle_mat_inv, angle_mat_inv], dim=0), torch.cat([series[:-1, :, :] - series[:-1, 0:1, :], series[1:, :, :] - series[:-1, 0:1, :]], dim=0).permute(0, 2, 1))
    trmat_ax_p0 = affine_matrix_from_points(p0p1[:len(angle_mat_inv)], p0p1[len(angle_mat_inv):])
    angle_ax_p0 = euler_from_matrix(trmat_ax_p0)

    dist_ax_p0_tr = trmat_ax_p0[:, :3, 3]

    dofs = torch.cat([dist_ax_p0_tr, angle_ax_p0], dim=-1)
    dofs = dofs.type(old_dtype)
    return dofs


def dof_to_series(start_point, dof):
    old_type = start_point.dtype
    start_point = start_point.type(torch.float64)
    dof = dof.type(torch.float64)

    b, t, _ = dof.shape
    dof = dof.view(b * t, -1)
    matrix = euler_matrix(dof[:, 3:])
    matrix[:, :3, 3] = dof[:, :3]
    matrix = matrix.view(b, t, 4, 4)

    start_axis = get_axis(start_point).permute(0, 2, 1)
    start_matrix = torch.cat([start_axis, start_point[:, 0, :].unsqueeze(-1)], dim=-1)
    start_matrix = F.pad(start_matrix, (0, 0, 0, 1))
    start_matrix[:, 3, 3] = 1
    start_matrix_inv = torch.inverse(start_matrix)

    matrix_chain = [start_matrix]
    for idx in range(matrix.shape[1]):
        matrix_chain.append(torch.bmm(matrix_chain[-1], matrix[:, idx]))
    matrix_chain = torch.stack(matrix_chain, dim=1)

    start_point_4d = F.pad(start_point, (0, 1))
    start_point_4d[:, :, 3] = 1
    series = torch.einsum('btij,bjk,bkl->btil', matrix_chain, start_matrix_inv, start_point_4d.permute(0, 2, 1)).permute(0, 1, 3, 2)[..., :3]

    series = series.type(old_type)
    return series


def series_to_mesh(series, height, width, origin=None):
    axis = get_axis(series).permute(0, 2, 1)

    if not hasattr(series_to_mesh, 'mesh') or height != series_to_mesh.height or width != series_to_mesh.width:
        range_x = torch.arange(-(height - 1) / 2, (height + 1) / 2, dtype=series.dtype, device=series.device)
        range_y = torch.arange(-(width - 1) / 2, (width + 1) / 2, dtype=series.dtype, device=series.device)
        mesh_x, mesh_y = torch.meshgrid(range_x, range_y, indexing='ij')
        mesh = torch.stack([mesh_y, -mesh_x, torch.zeros_like(mesh_x)], dim=-1)
        series_to_mesh.mesh = mesh
        series_to_mesh.height = height
        series_to_mesh.width = width

    center = series[:, 0, :].unsqueeze(1).unsqueeze(1)

    local_mesh = torch.einsum('Nij,HWj->NHWi', axis, series_to_mesh.mesh) + center
    if origin is not None:
        local_mesh = local_mesh + origin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return local_mesh


def is_in_ellipsoid(point, radius, keepdim=False):
    assert point.ndim == radius.ndim and radius.ndim in [1, 2]
    assert point.shape[-1] == radius.shape[-1] == 3
    return torch.norm(point / radius, 2, dim=-1, keepdim=keepdim) <= 1


def draw_sobol_normal_samples(d: int, n: int, dtype: torch.dtype, device: torch.device):
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=None)
    samples = engine.draw(n, dtype=dtype)
    v = 0.5 + (1 - torch.finfo(samples.dtype).eps) * (samples - 0.5)
    samples = torch.erfinv(2 * v - 1) * math.sqrt(2)
    return samples.to(device=device)


def sample_hypersphere(d: int, n: int, dtype: torch.dtype, device: torch.device):
    if d == 1:
        rnd = torch.randint(0, 2, (n, 1), dtype=dtype, device=device)
        return 2 * rnd - 1
    rnd = torch.randn(n, d, dtype=dtype, device=device)
    samples = rnd / torch.norm(rnd, dim=-1, keepdim=True)
    return samples


def sample_ellipsoid(n, radius=None):
    if not hasattr(sample_ellipsoid, 'L_inv'):
        assert isinstance(radius, torch.Tensor)
        C = 1 / radius ** 2 * torch.eye(3, dtype=torch.float64, device=radius.device)
        sample_ellipsoid.L_inv = torch.inverse(torch.linalg.cholesky(C)).unsqueeze(0)
    sample_ellipsoid.L_inv = sample_ellipsoid.L_inv.type(radius.dtype)

    theta = draw_sobol_normal_samples(d=3, n=n, dtype=radius.dtype, device=radius.device)
    theta = F.normalize(theta, 2, dim=-1)
    r = torch.rand(n, 1, dtype=radius.dtype, device=radius.device) ** (1 / 3)
    x = r * theta
    u = sample_ellipsoid.L_inv.expand(x.shape[0], 3, 3).bmm(x.unsqueeze(-1)).squeeze(-1)
    return u


def sample_points(n, height, width, radius):
    old_type = radius.dtype
    radius = radius.type(torch.float64)

    if not hasattr(sample_points, 'axis_x') or n != len(sample_points.axis_x):
        sample_points.axis_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=radius.dtype, device=radius.device).expand((n, 3))
        sample_points.axis_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=radius.dtype, device=radius.device).expand((n, 3))
    sample_points.axis_x = sample_points.axis_x.type(radius.dtype)
    sample_points.axis_y = sample_points.axis_y.type(radius.dtype)

    centers = sample_ellipsoid(n, radius=radius)
    ax = sample_hypersphere(d=3, n=n, dtype=radius.dtype, device=radius.device)

    plane_a = ax.cross(sample_points.axis_x, dim=-1)
    plane_a_ = ax.cross(sample_points.axis_y, dim=-1)
    flag_plane_a = torch.norm(plane_a, 1, dim=-1) == 0
    plane_a[flag_plane_a] = plane_a_[flag_plane_a]
    plane_a = F.normalize(plane_a, p=2, dim=-1)
    plane_b = ax.cross(plane_a, dim=-1)
    theta = torch.rand((n, 1), dtype=centers.dtype, device=centers.device) * 2 * np.pi
    ay = torch.cos(theta) * plane_a + torch.sin(theta) * plane_b

    h, w = ay * height / 2, ax * width / 2
    corner1s, corner2s = centers - h - w, centers - h + w

    return torch.stack([centers, corner1s, corner2s], dim=1).type(old_type)


def sample_points_at_border(n, height, width, radius, direct_down='-axis_y'):
    old_type = radius.dtype
    radius = radius.type(torch.float64)

    if not hasattr(sample_points_at_border, 'axis_x') or n != len(sample_points_at_border.axis_x):
        sample_points_at_border.axis_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=radius.dtype, device=radius.device).expand((n, 3))
        sample_points_at_border.axis_y = torch.tensor([[0.0, 1.0, 0.0]], dtype=radius.dtype, device=radius.device).expand((n, 3))
        sample_points_at_border.axis_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=radius.dtype, device=radius.device).expand((n, 3))
    sample_points_at_border.axis_x = sample_points_at_border.axis_x.type(radius.dtype)
    sample_points_at_border.axis_y = sample_points_at_border.axis_y.type(radius.dtype)
    sample_points_at_border.axis_z = sample_points_at_border.axis_z.type(radius.dtype)

    if direct_down.startswith('-'):
        direct_down = direct_down[1:]
        direct_down_coefficient = -1
    else:
        direct_down_coefficient = 1

    centers = sample_ellipsoid(n, radius)
    dist_flag = is_in_ellipsoid(centers, radius * 0.8, keepdim=True) | (torch.abs(centers[:, 1:2]) > 0.6 * radius[0, 1])
    flag_n = torch.sum(dist_flag)
    while flag_n > 0:
        centers_ = sample_ellipsoid(flag_n, radius)
        centers[dist_flag.squeeze(-1), :] = centers_
        dist_flag = is_in_ellipsoid(centers, radius * 0.8, keepdim=True) | (torch.abs(centers[:, 1:2]) > 0.6 * radius[0, 1])
        flag_n = torch.sum(dist_flag)

    direct_down = direct_down_coefficient * getattr(sample_points_at_border, direct_down)
    direct_down = sample_points_by_limit(direct_down, min_cos=0.94, n=1, dtype=radius.dtype, device=radius.device).squeeze(1)
    normals = sample_points_by_limit(-centers, min_cos=0.984, n=1, dtype=radius.dtype, device=radius.device).squeeze(1)
    normals = normals - torch.sum(normals * direct_down, dim=-1, keepdim=True) * direct_down
    normals = F.normalize(normals, p=2, dim=-1)
    direct_plane = normals.cross(direct_down, dim=-1)

    h, w = direct_down * height / 2, direct_plane * width / 2
    corner1s, corner2s = centers + h - w, centers + h + w

    return torch.stack([centers, corner1s, corner2s], dim=1).type(old_type)


def sample_points_by_limit(normals, min_cos=-1.0, max_cos=1.0, n=1, dtype=torch.float32, device='cuda'):
    assert normals.ndim == 2
    assert not (torch.norm(normals, 1, dim=-1) == 0).any()

    A0 = normals
    n0, n1, n2 = torch.split(normals, 1, dim=-1)
    n0n1, n1n2, n0n2 = n0 * n1, n1 * n2, n0 * n2
    s0, s1, s2 = torch.split(normals ** 2, 1, dim=-1)
    zero = torch.zeros_like(n0)
    A1 = torch.cat([-n1, n0 - n2, n1], dim=-1)
    A1_exp = torch.cat([n2, zero, -n0], dim=-1)
    A1_zero = torch.norm(A1, 1, dim=-1) == 0
    A1[A1_zero] = A1_exp[A1_zero]
    A2 = torch.cat([s1 + s2 - n0n2, -n0n1 - n1n2, s0 + s1 - n0n2], dim=-1)
    A2_exp = torch.cat([n1, -n0, zero], dim=-1)
    A2_zero = torch.norm(A2, 1, dim=-1) == 0
    A2[A2_zero] = A2_exp[A2_zero]

    A = torch.stack([A0, A1, A2], dim=1)
    A = A / torch.norm(A, 2, dim=-1, keepdim=True)
    A = A.permute(0, 2, 1)

    r = torch.rand((normals.shape[0], 2, n), dtype=dtype, device=device)
    theta, t = torch.split(r, 1, dim=1)
    min_theta = np.arccos(max_cos)
    max_theta = np.arccos(min_cos)
    theta = (1 - theta) * (max_theta - min_theta) + min_theta
    length_shallow, radius = torch.cos(theta), torch.sin(theta)
    t = t * np.pi * 2
    cos, sin = radius * torch.cos(t), radius * torch.sin(t)
    mat = torch.cat([length_shallow, cos, sin], dim=1)

    result = torch.bmm(A, mat).permute(0, 2, 1)
    result = result / torch.norm(result, dim=-1, keepdim=True)

    return result


def get_slices(volume, series, height, width, volume_origin):
    mesh = series_to_mesh(series, height=height, width=width, origin=volume_origin)
    volume_size = torch.tensor(volume.shape, dtype=mesh.dtype, device=mesh.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mesh = mesh.unsqueeze(0) / volume_size
    mesh = (mesh * 2 - 1).flip(-1)
    slices = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), mesh, mode='bilinear', padding_mode='border', align_corners=False)
    slices = slices.squeeze(0).squeeze(0)
    return slices


def solve_quadratic_equation(a, b, c):
    delta = b ** 2 - 4 * a * c
    assert delta >= 0.0, delta
    if a == 0.0:
        return -c / b
    theta = delta.sqrt() if isinstance(a, (torch.Tensor, np.ndarray)) else math.sqrt(delta)
    a2 = 2 * a
    x1, x2 = (-b + theta) / a2, (-b - theta) / a2
    mi, ma = min(x1, x2), max(x1, x2)
    if mi >= 0:
        return mi
    if ma >= 0:
        return ma
    return None


def update_dva(d, v, a):
    flag = False
    if d < 0:
        d, v, a = -d, -v, -a
        flag = True
    _a, _b, _c = a / 2, v, -d
    nd, nv, na = d, v, a
    d = _b ** 2 / (4 * _c)
    if _a < d:
        na = np.random.rand() * d * 2
    if flag:
        nd, nv, na = -nd, -nv, -na
    return nd, nv, na


def sample_line(start_points, end_point, start_velocity, start_accele, radius, n=None, min_n=None):
    old_dtype = start_points.dtype
    start_points = start_points.type(torch.float64)
    end_point = end_point.type(torch.float64)
    start_velocity = start_velocity.type(torch.float64) if isinstance(start_velocity, torch.Tensor) else start_velocity
    start_accele = start_accele.type(torch.float64) if isinstance(start_accele, torch.Tensor) else start_accele
    radius = radius.type(torch.float64)

    gap = end_point - start_points[0, :]
    gap0l = torch.norm(gap, 2, dim=-1, keepdim=True)
    gap0d = gap / gap0l

    sv0 = torch.sum(start_velocity * gap0d) * gap0d
    sv1 = start_velocity - sv0
    sa0 = torch.sum(start_accele * gap0d) * gap0d
    sa1 = start_accele - sa0

    _, _, nsa0 = update_dva(gap0l, (sv0 / gap0d)[0], (sa0 / gap0d)[0])
    sa0 = nsa0 * gap0d
    t = solve_quadratic_equation((sa0 / gap0d)[0] / 2, (sv0 / gap0d)[0], -gap0l)
    b = -(6 * sv1 + 3 * sa1 * t) / (t ** 2)

    n = n or max(int(torch.round(t)), 1)
    if min_n is not None and n < min_n:
        return None
    tgap = t / n

    idx = torch.arange(1, n + 1, dtype=start_points.dtype, device=start_points.device).unsqueeze(-1)
    tidx = idx * tgap
    tidx = tidx + 0.05 * tgap * torch.randn_like(tidx)
    t2 = tidx ** 2
    t3 = tidx * t2

    s0 = sv0 * tidx + sa0 * t2 / 2
    s1 = sv1 * tidx + sa1 * t2 / 2 + b * t3 / 6
    s = s0 + s1

    series = start_points.unsqueeze(0) + s.unsqueeze(1)
    length_velocity = torch.norm(series - torch.cat([start_points.unsqueeze(0), series[:-1, :, :]], dim=0), 2, dim=[1, 2])
    length_rand = torch.randn((series.shape[0], 1, 3), dtype=series.dtype, device=series.device)
    series = series + (0.01 * length_velocity.unsqueeze(-1).unsqueeze(-1) * length_rand).expand(series.shape[0], 3, 3)

    flag = is_in_ellipsoid(series[:, 0, :], radius)
    flag = ~(torch.cumsum(~flag, dim=-1) > 0)
    series = series[flag, ...]
    if min_n is not None and len(series) < min_n:
        return None

    series = series.type(old_dtype)
    return series


def sample_sector(start_points, sector_point, end_direct, max_velocity, start_accele, end_accele, radius, n=None, min_n=None):
    old_dtype = start_points.dtype
    start_points = start_points.type(torch.float64)
    sector_point = sector_point.type(torch.float64)
    end_direct = end_direct.type(torch.float64) if isinstance(end_direct, torch.Tensor) else end_direct
    max_velocity = max_velocity.type(torch.float64) if isinstance(max_velocity, torch.Tensor) else max_velocity
    start_accele = start_accele.type(torch.float64) if isinstance(start_accele, torch.Tensor) else start_accele
    end_accele = end_accele.type(torch.float64) if isinstance(end_accele, torch.Tensor) else end_accele
    radius = radius.type(torch.float64)

    start_direct = start_points[0, :] - sector_point
    start_direct = start_direct / torch.norm(start_direct, 2)
    end_direct = end_direct / torch.norm(end_direct, 2)
    sector_direct = torch.cross(start_direct, end_direct, dim=-1)
    sector_direct = sector_direct / torch.norm(sector_direct, 2)
    cos = torch.sum(start_direct * end_direct)
    cos.clamp_(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    angle = torch.acos(cos).item()

    ts = 0
    gap_s = start_accele * ts ** 2 / 2
    te = 0
    gap_e = -end_accele * te ** 2 / 2
    gap_med = angle - gap_s - gap_e
    tm = gap_med / max_velocity

    if tm < 0:
        ts, te = 0, 0
        tm = angle / max_velocity
        gap_s, gap_e = 0, 0
        gap_med = angle

    t = ts + tm + te
    n = n or max(int(round(t)), 1)
    if min_n is not None and n < min_n:
        return None
    tgap = t / n

    idx = torch.arange(1, n + 1, dtype=start_points.dtype, device=start_points.device).unsqueeze(-1)
    tidx = idx * tgap
    tidx = tidx + 0.05 * tgap * torch.randn_like(tidx)

    flag_1 = tidx <= ts
    flag_2 = ~flag_1 & (tidx <= ts + tm)
    flag_3 = ~(flag_1 | flag_2)

    s1 = start_accele * tidx[flag_1] ** 2 / 2
    s2 = gap_s + max_velocity * (tidx[flag_2] - ts)
    tt = tidx[flag_3] - ts - tm
    s3 = gap_s + gap_med + max_velocity * tt + end_accele * tt ** 2 / 2
    s = torch.cat([s1, s2, s3], dim=-1)
    s = s + 0.1 * torch.cat([s[1:] - s[:-1], s[0:1]], dim=-1) * torch.randn_like(s)

    s_ = s.unsqueeze(-1)
    s_sin = torch.sin(s_ / 2)
    s_cos = torch.cos(s_ / 2)
    series = quaternion_rotation_mul_theta(start_points, sector_point.unsqueeze(0), sector_direct, s_sin, s_cos)

    flag = is_in_ellipsoid(series[:, 0, :], radius)
    flag = ~(torch.cumsum(~flag, dim=-1) > 0)
    series = series[flag, ...]
    if min_n is not None and len(series) < min_n:
        return None

    series = series.type(old_dtype)
    return series
