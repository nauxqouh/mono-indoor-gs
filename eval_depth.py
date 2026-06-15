import os
import cv2
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -- Load depths helper --------------------------------------------------------
def load_gt_depths(image_list, datadir, H=None, W=None):
    depths = []
    masks = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{:05d}.png'.format(int(frame_id)))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"GT depth not found: {depth_path}")
        depth = depth.astype(np.float32) / 1000.0
        
        if H is not None:
            mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            masks.append(depth > 0)
    return np.stack(depths), np.stack(masks)

def load_depths_tiff(image_list, datadir, H=None, W=None):
    depths = []

    for i, image_name in enumerate(image_list):
        depth_path = os.path.join(datadir, 'depth_{:05d}.tiff'.format(i))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Pred .tiff depth not found: {depth_path}")
        depth = depth.astype(np.float32)
        
        if H is not None:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    
        depths.append(depth)
        
    return np.stack(depths)

def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25      ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def depth_evaluation(
    rgb_dir, gt_dir, pred_dir, split="test", llffhold=8, 
    min_depth=0.1, max_depth=20.0, scale_depth=True,
    save_dir=None
):
    # train/test mapping
    all_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
    if split.lower() == "test":
        image_list = [img for idx, img in enumerate(all_images) if idx % llffhold == 0]
    else:
        image_list = [img for idx, img in enumerate(all_images) if idx % llffhold != 0]

    logger.info(f"Loading {split.upper()} - {len(image_list)} frames")

    gt_batch, gt_masks = load_gt_depths(image_list, gt_dir)
    H, W = gt_batch.shape[1], gt_batch.shape[2]
    pred_batch = load_depths_tiff(image_list, pred_dir, H=H, W=W)

    gt_valid_list = []
    pred_valid_list = []

    for i in range(len(image_list)):
        gt = gt_batch[i]
        pred = pred_batch[i]
        
        mask = (gt > min_depth) & (gt < max_depth) & (pred > 0)
        if mask.sum() == 0:
            continue
            
        gt_valid_list.append(gt[mask])
        pred_valid_list.append(pred[mask])

    if len(gt_valid_list) == 0:
        logger.warning("No valid to evaluate!")
        return None

    # Global Median Scale
    ratio = 1.0
    if scale_depth:
        global_gt_median = np.median(np.concatenate(gt_valid_list))
        global_pred_median = np.median(np.concatenate(pred_valid_list))
        ratio = global_gt_median / (global_pred_median + 1e-6)
        logger.info(f"Median Scale Ratio (GT / Pred): {ratio:.4f}")
        
    # Scale, Clamp and compute Errors
    errors = []
    for i in range(len(pred_valid_list)):
        gt_valid = gt_valid_list[i]
        pred_valid = pred_valid_list[i]
        
        pred_scaled = pred_valid * ratio
        pred_scaled = np.clip(pred_scaled, min_depth, max_depth)
        errors.append(compute_errors(gt_valid, pred_scaled))

    # Calculate mean errors
    mean_errors = np.array(errors).mean(axis=0)
    
    # Logging Results
    logger.info("RESULTS")
    logger.info(("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    logger.info("-" * 80)
    logger.info(("{:10.4f} | " * 7).format(*mean_errors.tolist()))
    
    # File Saving Logic
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'depth_evaluation.txt'), 'a+') as f:
            f.seek(0)
            lines = f.readlines()
            if len(lines) == 0:
                f.write(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '    scale_depth\n')
            f.seek(0, 2)
            f.write(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + f"    {scale_depth}   \\\\\n")
            
        logger.info(f"Saved results in: {os.path.join(save_dir, 'depth_evaluation.txt')}")
        
    return mean_errors