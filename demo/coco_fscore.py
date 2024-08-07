from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

iou = 0.5
gt_path = "/mnt/nvstorage/gis/seedling/data/tiled_coco/2560/anno/test.json"
pred_path = "/home/sema/sahi/runs/predict/exp4/result.json"
plot_path = "/home/sema/sahi/runs/predict/exp4/plot.png"
cocoGt = COCO(gt_path)  # initialize COCO ground truth api
cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO prediction api
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")  # initialize COCO evaluation api
cocoEval.evaluate()
cocoEval.accumulateFBeta()
print(cocoEval.getBestFBeta(beta=1, iouThr=iou, classIdx=0, average="macro"))
print(cocoEval.getBestFBeta(beta=1, iouThr=iou, classIdx=1, average="macro"))
cocoEval.plotFBetaCurve(plot_path.replace(".png", "_0.png"), betas=[1], classIdx=0, iouThr=iou, average="macro")
cocoEval.plotFBetaCurve(plot_path.replace(".png", "_1.png"), betas=[1], classIdx=1, iouThr=iou, average="macro")
cocoEval.accumulate()
cocoEval.summarize()
