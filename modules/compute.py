import numpy as np


def computeOks(self, imgId, catId):
    p = self.params
    # dimention here should be Nxm
    gts = self._gts[imgId, catId]
    dts = self._dts[imgId, catId]
    inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
    dts = [dts[i] for i in inds]
    if len(dts) > p.maxDets[-1]:
        dts = dts[0 : p.maxDets[-1]]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0
    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt["keypoints"])
        xg = g[0::3]
        yg = g[1::3]
        vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt["bbox"]
        x0 = bb[0] - bb[2]
        x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]
        y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt["keypoints"])
            xd = d[0::3]
            yd = d[1::3]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
            e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
            if k1 > 0:
                e = e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]

    return ious
