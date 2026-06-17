# borrowed from nerfingmvs and neuralreon

import os, cv2,logging
import numpy as np
import open3d as o3d

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils

def evaluate_geometry_neucon(file_pred, file_trgt, threshold=.05, down_sample=.02):
    """ Borrowed from NeuralRecon
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    def nn_correspondance(verts1, verts2):
        """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's

        Returns:
            ([indices], [distances])

        """

        indices = []
        distances = []
        if len(verts1) == 0 or len(verts2) == 0:
            return indices, distances

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts1)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        for vert in verts2:
            _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
            indices.append(inds[0])
            distances.append(np.sqrt(dist[0]))

        return indices, distances

    pcd_pred = GeoUtils.read_point_cloud(file_pred)
    pcd_trgt = GeoUtils.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),  # pred->gt
               'dist2': np.mean(dist1),  # gt -> pred
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    # plot graph
    # if path_fscore_curve:
    #     EvalUtils.draw_figure_fscore(path_fscore_curve, threshold, dist2, dist1, plot_stretch=5)

    metrics = np.array([np.mean(dist2), np.mean(dist1), precision, recal, fscore])
    logging.info(f'{file_pred.split("/")[-1]}: {metrics}')
    return metrics

def evaluate_3D_mesh(path_mesh_pred, scene_name, dir_dataset = './dataset/indoor',
                            eval_threshold = 0.05, reso_level = 2.0, 
                            check_existence = True):
    '''Evaluate geometry quality of neus using Precison, Recall and F-score.
    '''
    path_intrin = f'{dir_dataset}/intrinsic_depth.txt'
    target_img_size = (640, 480)
    dir_scan = f'{dir_dataset}/{scene_name}'
    dir_poses = f'{dir_scan}/pose'
    # dir_images = f'{dir_scan}/image'
    
    path_mesh_gt = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2.ply'
    path_mesh_gt_clean = IOUtils.add_file_name_suffix(path_mesh_gt, '_clean')
    path_mesh_gt_2dmask = f'{dir_dataset}/{scene_name}/{scene_name}_vh_clean_2_2dmask.npz'
    
    # (1) clean GT mesh
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    GeoUtils.generate_mesh_2dmask(path_mesh_gt_2dmask, path_mesh_gt_clean, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                check_existence = check_existence)
    # for fair comparison
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_gt_clean, path_mesh_gt, 
                                                path_intrin, dir_poses, 
                                                target_img_size, reso_level=reso_level,
                                                path_mask_npz=path_mesh_gt_2dmask,
                                                check_existence = check_existence)


    # (2) clean predicted mesh
    path_mesh_pred_clean_bbox = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox')
    path_mesh_pred_clean_bbox_faces = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces')
    path_mesh_pred_clean_bbox_faces_mask = IOUtils.add_file_name_suffix(path_mesh_pred, '_clean_bbox_faces_mask')

    GeoUtils.clean_mesh_points_outside_bbox(path_mesh_pred_clean_bbox, path_mesh_pred, path_mesh_gt,
                                                scale_bbox=1.1,
                                                check_existence = check_existence)
    GeoUtils.clean_mesh_faces_outside_frustum(path_mesh_pred_clean_bbox_faces, path_mesh_pred_clean_bbox, 
                                                    path_intrin, dir_poses, 
                                                    target_img_size, reso_level=reso_level,
                                                    check_existence = check_existence)
    GeoUtils.clean_mesh_points_outside_frustum(path_mesh_pred_clean_bbox_faces_mask, path_mesh_pred_clean_bbox_faces, 
                                                    path_intrin, dir_poses, 
                                                    target_img_size, reso_level=reso_level,
                                                    path_mask_npz=path_mesh_gt_2dmask,
                                                    check_existence = check_existence)
    
    path_eval = path_mesh_pred_clean_bbox_faces_mask 
    metrices_eval = evaluate_geometry_neucon(path_eval, path_mesh_gt_clean, 
                                                        threshold=eval_threshold, down_sample=.02) #f'{dir_eval_fig}/{scene_name}_step{iter_step:06d}_thres{eval_threshold}.png')

    return metrices_eval

def save_evaluation_results_to_latex(path_log, 
                                        header = '                     Accu.      Comp.      Prec.     Recall     F-score \n', 
                                        results = None, 
                                        names_item = None, 
                                        save_mean = None, 
                                        mode = 'w',
                                        precision = 3,
                                        eval_log = None):
    '''Save evaluation results to txt in latex mode
    Args:
        header:
            for F-score: '                     Accu.      Comp.      Prec.     Recall     F-score \n'
        results:
            narray, N*M, N lines with M metrics
        names_item:
            N*1, item name for each line
        save_mean: 
            whether calculate the mean value for each metric
        mode:
            write mode, default 'w'
    '''
    # save evaluation results to latex format
    with open(path_log, mode) as f_log:
        if header:
            f_log.writelines(header)
        if results is not None:
            num_lines, num_metrices = results.shape
            if names_item is None:
                names_item = np.arange(results.shape[0])
            for idx in range(num_lines):
                f_log.writelines((f'{names_item[idx]}    ' + ("&{: 8.3f}  " * num_metrices).format(*results[idx, :].tolist())) + " \\\ \n")
                eval_log((f'{names_item[idx]}    ' + ("&{: 8.3f}  " * num_metrices).format(*results[idx, :].tolist())) + " \\\ \n", "./0-log/exp_results.txt")
                eval_log((f'{names_item[idx]}    ' + ("&{: 8.3f}  " * num_metrices).format(*results[idx, :].tolist())) + " \\\ \n", "./0-log/log.txt")
        if save_mean:
            mean_results = results.mean(axis=0)     # 4*7
            mean_results = np.round(mean_results, decimals=precision)
            f_log.writelines(( '       Mean    ' + " &{: 8.3f} " * num_metrices).format(*mean_results[:].tolist()) + " \\\ \n")
 

