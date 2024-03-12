import numpy as np
import mindspore
import x2ms_adapter


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        # anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        # boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)
        anchors[:, 3:6] = mindspore.ops.clamp(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = mindspore.ops.clamp(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = x2ms_adapter.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = x2ms_adapter.split(boxes, 1, dim=-1)

        diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = x2ms_adapter.log(dxg / dxa)
        dyt = x2ms_adapter.log(dyg / dya)
        dzt = x2ms_adapter.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = x2ms_adapter.cos(rg) - x2ms_adapter.cos(ra)
            rt_sin = x2ms_adapter.sin(rg) - x2ms_adapter.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return x2ms_adapter.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = x2ms_adapter.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = x2ms_adapter.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = x2ms_adapter.split(box_encodings, 1, dim=-1)

        diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = x2ms_adapter.exp(dxt) * dxa
        dyg = x2ms_adapter.exp(dyt) * dya
        dzg = x2ms_adapter.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + x2ms_adapter.cos(ra)
            rg_sin = sint + x2ms_adapter.sin(ra)
            rg = x2ms_adapter.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return x2ms_adapter.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = x2ms_adapter.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = x2ms_adapter.split(box_encodings, 1, dim=-1)

        diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = x2ms_adapter.exp(lt) * dxa
        dyg = x2ms_adapter.exp(wt) * dya
        dzg = x2ms_adapter.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return x2ms_adapter.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = x2ms_adapter.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = x2ms_adapter.split(box_encodings, 1, dim=-1)

        diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = x2ms_adapter.exp(lt) * dxa
        dyg = x2ms_adapter.exp(wt) * dya
        dzg = x2ms_adapter.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return x2ms_adapter.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(np.array(kwargs['mean_size'])))
            assert x2ms_adapter.tensor_api.x2ms_min(self.mean_size) > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        # gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)
        gt_boxes[:, 3:6] = mindspore.ops.clamp(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = x2ms_adapter.split(gt_boxes, 1, dim=-1)
        xa, ya, za = x2ms_adapter.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert x2ms_adapter.tensor_api.x2ms_max(gt_classes) <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = x2ms_adapter.split(point_anchor_size, 1, dim=-1)
            diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = x2ms_adapter.log(dxg / dxa)
            dyt = x2ms_adapter.log(dyg / dya)
            dzt = x2ms_adapter.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = x2ms_adapter.log(dxg)
            dyt = x2ms_adapter.log(dyg)
            dzt = x2ms_adapter.log(dzg)

        cts = [g for g in cgs]
        return x2ms_adapter.cat([xt, yt, zt, dxt, dyt, dzt, x2ms_adapter.cos(rg), x2ms_adapter.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = x2ms_adapter.split(box_encodings, 1, dim=-1)
        xa, ya, za = x2ms_adapter.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert x2ms_adapter.tensor_api.x2ms_max(pred_classes) <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = x2ms_adapter.split(point_anchor_size, 1, dim=-1)
            diagonal = x2ms_adapter.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = x2ms_adapter.exp(dxt) * dxa
            dyg = x2ms_adapter.exp(dyt) * dya
            dzg = x2ms_adapter.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = x2ms_adapter.split(x2ms_adapter.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = x2ms_adapter.atan2(sint, cost)

        cgs = [t for t in cts]
        return x2ms_adapter.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
