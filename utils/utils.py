import os
import numpy as np


def LBS(x, W1, T, R):
    bx = (T @ x.T).permute(0, 3, 1, 2)
    wbx = W1[None, :, :, None] * bx
    wbx = wbx.permute(0, 2, 1, 3)
    wbx = wbx.sum(1, keepdim=True)
    wbx = (R @ (wbx[:, 0].permute(0, 2, 1))).permute(0, 2, 1)

    return wbx

def LBS_notrans(x, W1, T):

    final_wbx = torch.zeros_like(x, requires_grad=True, device="cuda")
    sum_weight = torch.zeros_like(W1[None, :, [0]])
    for b_ind in range(T.shape[1]):
        final_wbx = final_wbx + T[:, [b_ind]].act(x) * W1[None, :, [b_ind]]
        sum_weight += W1[None, :, [b_ind]]
    return final_wbx



### for flow


UNKNOWN_FLOW_THRESH = 1e7
# SMALLFLOW = 0.0
# LARGEFLOW = 1e8

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

### plot

def per_frame_energy_plot(color, mask, flow, rendering_color, rendering_mask, rendering_flow, flow_mask, wbx, points_info,
                          face_info, epoch_id, iter_id, B_size, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lap_loss = Preframe_LaplacianLoss(points_info, face_info)
    arap_loss = Preframe_ARAPLoss(points_info, face_info)
    loss_color = F.mse_loss(color, rendering_color, reduction='none').mean((1, 2, 3))
    loss_mask = F.mse_loss(mask, rendering_mask, reduction='none').mean((1, 2))
    loss_flow = F.mse_loss(flow[:-1] * flow_mask, rendering_flow * flow_mask, reduction='none').mean((1, 2, 3))

    loss_smooth_all = lap_loss(wbx[:, :, :-1])
    loss_arap = 0
    smooth_path, arap_path = os.path.join(out_path, 'smooth', 'Epoch{}'.format(epoch_id)), os.path.join(out_path, 'arap',
                                                                                                'Epoch{}'.format(
                                                                                                    epoch_id))
    if not os.path.exists(smooth_path):
        os.makedirs(smooth_path)
    if not os.path.exists(arap_path):
        os.makedirs(arap_path)
    for ind in range(loss_smooth_all.shape[0]):
        loss_arap = arap_loss(wbx[ind:ind + 1, :, :-1], basic_mesh[None][:, :, :-1])
        loss_smooth_perF = loss_smooth_all[ind]
        np.save(os.path.join(smooth_path, '{:04d}'.format(ind + iter_id * B_size)),
                loss_smooth_perF.detach().cpu().numpy())
        np.save(os.path.join(arap_path, '{:04d}'.format(ind + iter_id * B_size)), loss_arap.detach().cpu().numpy())
    del loss_smooth_all, loss_color, loss_mask, loss_flow, loss_arap, loss_smooth_perF
    return


def distance_matrix(centers):
    X = centers.T
    m, n = X.shape
    G = np.dot(X.T, X)
    D = np.zeros([n, n])

    for i in range(n):
        D[i, i] = 100
        for j in range(i+1, n):
            d = X[:, i] - X[:, j]
            D[i,j] = G[i,i] - 2 * G[i,j] + G[j,j]
            D[j,i] = D[i,j]
    return D
