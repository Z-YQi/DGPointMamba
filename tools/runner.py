import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import torch.nn.functional as F
import open3d as o3d
from tools import builder
from utils import dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.realtime_render import *
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2,UnidirectionalChamferDistance,DirectedHausdorffDistance
from utils import loss_util
from utils.AverageMeter import AverageMeter
from utils.domain_generators import build_domain_generator
from utils.experiment_logging import ExperimentLogger, metric_dict


DG_METRIC_NAMES = ['F-Score', 'CDL1', 'CDL2', 'EMDistance', 'UCD', 'UHD']


def _ensure_point_cloud_batch(points, name):
    if points.dim() == 4 and points.size(0) == 1:
        points = points.squeeze(0)
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError(f"{name} must have shape [B, N, 3], got {tuple(points.shape)}")
    return points.float().contiguous()


def _move_to_device(tensor, args):
    tensor = tensor.float()
    if args.use_gpu:
        tensor = tensor.cuda(non_blocking=True)
    return tensor


def _model_rebuild_points(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def _scalar_item(value):
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _format_generator_stats(stats):
    return (
        f"A={stats['sinpoint_A']:.4f}, w={stats['sinpoint_w']:.4f}, "
        f"rand_center_num={stats['sinpoint_rand_center_num']}, sample={stats['sinpoint_sample']}, "
        f"mean_delta_p={stats['mean_delta_p']:.6f}, max_delta_p={stats['max_delta_p']:.6f}"
    )


def _loss_weight(config, key, default):
    loss_weights = config.get("loss_weights", None)
    if loss_weights is None:
        return default
    return float(loss_weights.get(key, default))


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _format_metric_header(metric_names):
    metric_names = list(metric_names)
    if metric_names != DG_METRIC_NAMES:
        raise ValueError(f"Metric order mismatch: expected {DG_METRIC_NAMES}, got {metric_names}")
    return ' | '.join(metric_names)


def _format_metric_values(metric_values):
    return ' | '.join('%.6f' % float(value) for value in metric_values)


def _visualization_output_dir(args, config):
    logging_config = _cfg_get(config, "logging", None)
    visualization_dir = _cfg_get(logging_config, "visualization_dir", "point_visualization")
    if os.path.isabs(visualization_dir):
        return visualization_dir
    return os.path.join(getattr(args, "experiment_path", "."), visualization_dir)


def _write_test_experiment_summary(experiment_logger, args, metrics_state):
    if experiment_logger is None:
        return
    checkpoint_path = getattr(args, "ckpts", "") or ""
    experiment_logger.log_metrics({
        'event': 'test',
        **metrics_state,
    })
    experiment_logger.update_best('', metrics_state, checkpoint_path)
    experiment_logger.write_summary(status='completed', metrics=metrics_state, checkpoint_path=checkpoint_path)
    experiment_logger.write_run_meta(status='completed')



def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UCD_distance = UnidirectionalChamferDistance()
    UHD_distance = DirectedHausdorffDistance()
    completion_loss = loss_util.Completionloss(loss_func=config.consider_metric)
    # Source-only DG training: use CRN/ShapeNet train data only.
    train_sampler, train_dataloader = builder.virtual_dataset_builder(args, config.dataset.train)
    config.dataset.test._base_.SPLIT = 'test'
    _, real_test_dataloader = builder.real_dataset_builder(args, config.dataset.test)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    domain_generator = build_domain_generator(config)
    if domain_generator is None:
        raise ValueError("Fixed SinPoint baseline requires domain_generator.type: fixed_sinpoint")
    lambda_aug = _loss_weight(config, "aug", 1.0)
    generator_device = next(base_model.parameters()).device
    init_num_points = max(getattr(domain_generator, "rand_center_num", 4), 4)
    _, init_generator_stats = domain_generator(torch.randn(1, init_num_points, 3, device=generator_device))
    print_log(f'[DGPointMamba] Domain generator initialized: fixed_sinpoint ({_format_generator_stats(init_generator_stats)})', logger=logger)
    print_log(f'[DGPointMamba] Training loss: loss_rec_clean + {lambda_aug:.4f} * loss_rec_aug_src', logger=logger)
    experiment_logger = ExperimentLogger(args, config)
    experiment_logger.write_run_meta(status='running')
    experiment_logger.write_summary(status='running')

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_meter_loss = AverageMeter(['loss_total', 'loss_rec_clean', 'loss_rec_aug_src'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        source_dataset_name = config.dataset.train._base_.NAME
        target_dataset_name = config.dataset.train.real_dataset
        if source_dataset_name != 'CRNShapeNet':
            raise NotImplementedError(f'Train phase does not support source dataset {source_dataset_name}')

        for idx, source_data in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            num_iter += 1
            n_itr = epoch * n_batches + idx

            source_gt, source_partial, _ = source_data
            source_gt = _ensure_point_cloud_batch(_move_to_device(source_gt, args), "Y_s")
            source_partial = _ensure_point_cloud_batch(_move_to_device(source_partial, args), "P_s")

            if target_dataset_name in ['3D_FUTURE', 'ModelNet'] and config.dataset.train._base_.CLASS_CHOICE in ['lamp', 'table']:
                source_partial, _, _, _ = partial_render_batch(source_gt, source_partial)
                source_partial = _ensure_point_cloud_batch(source_partial, "P_s")

            aug_partial, generator_stats = domain_generator(source_partial)
            assert aug_partial.device == source_partial.device, "P_aug.device must match P_s.device"
            assert aug_partial.shape == source_partial.shape, f"P_aug.shape must match P_s.shape: {aug_partial.shape} vs {source_partial.shape}"
            if not generator_stats.get("stats_finite", False):
                raise FloatingPointError(f"Fixed SinPoint produced non-finite stats: {generator_stats}")

            pred_clean = _model_rebuild_points(base_model(source_partial))
            loss_rec_clean, _ = completion_loss.get_loss(pred_clean, source_partial, source_gt)

            pred_aug = _model_rebuild_points(base_model(aug_partial))
            loss_rec_aug_src, _ = completion_loss.get_loss(
                pred_aug,
                None,
                source_gt,
                include_partial_matching=False,
            )

            loss_total = loss_rec_clean + lambda_aug * loss_rec_aug_src
            if loss_total.dim() != 0:
                loss_total = loss_total.mean()
            loss_total.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_total_log = _scalar_item(dist_utils.reduce_tensor(loss_total.detach(), args))
                loss_rec_clean_log = _scalar_item(dist_utils.reduce_tensor(loss_rec_clean.detach(), args))
                loss_rec_aug_log = _scalar_item(dist_utils.reduce_tensor(loss_rec_aug_src.detach(), args))
            else:
                loss_total_log = _scalar_item(loss_total)
                loss_rec_clean_log = _scalar_item(loss_rec_clean)
                loss_rec_aug_log = _scalar_item(loss_rec_aug_src)
            avg_meter_loss.update([loss_total_log, loss_rec_clean_log, loss_rec_aug_log])

            if train_writer is not None and args.local_rank == 0:
                train_writer.add_scalar('DGPointMamba/Train/loss_total', loss_total_log, n_itr)
                train_writer.add_scalar('DGPointMamba/Train/loss_rec_clean', loss_rec_clean_log, n_itr)
                train_writer.add_scalar('DGPointMamba/Train/loss_rec_aug_src', loss_rec_aug_log, n_itr)
                train_writer.add_scalar('DGPointMamba/Generator/FixedSinPoint/mean_delta_p', generator_stats['mean_delta_p'], n_itr)
                train_writer.add_scalar('DGPointMamba/Generator/FixedSinPoint/max_delta_p', generator_stats['max_delta_p'], n_itr)

            if args.local_rank == 0:
                experiment_logger.log_metrics({
                    'event': 'train_step',
                    'epoch': epoch,
                    'step': n_itr,
                    'batch': idx,
                    'loss_total': loss_total_log,
                    'loss_rec_clean': loss_rec_clean_log,
                    'loss_rec_aug_src': loss_rec_aug_log,
                    **generator_stats,
                })
                

            if args.distributed:
                torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:

                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f | loss_total = %.6f | loss_rec_clean = %.6f | loss_rec_aug_src = %.6f | generator = %s' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                        ['%.4f' % l for l in avg_meter_loss.val()], optimizer.param_groups[0]['lr'], loss_total_log,
                        loss_rec_clean_log, loss_rec_aug_log, _format_generator_stats(generator_stats)), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()


        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in avg_meter_loss.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
             # Validate the current model
            metrics = validate(base_model, real_test_dataloader, epoch, 
                                    ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger=logger)
            metrics_state = metrics.state_dict()
            if args.local_rank == 0:
                experiment_logger.log_metrics({
                    'event': 'validation',
                    'epoch': epoch,
                    **metrics_state,
                })
        
             # Save ckeckpoints
            is_best = metrics.better_than(best_metrics)
            if is_best:
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt_dg_best', args, logger = logger)
                    if args.local_rank == 0:
                        experiment_logger.update_best(
                            epoch,
                            metrics_state,
                            os.path.join(args.experiment_path, 'ckpt_dg_best.pth'),
                        )
            if args.local_rank == 0:
                experiment_logger.write_summary(status='running')
                experiment_logger.write_run_meta(status='running')
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)

    if args.local_rank == 0:
        experiment_logger.write_summary(status='completed')
        experiment_logger.write_run_meta(status='completed')

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model,real_test_dataloader, epoch, ChamferDisL1, ChamferDisL2, UCD_distance, UHD_distance, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    test_losses = AverageMeter(['Rebuild_Loss_L1','Rebuild_Loss_L2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(real_test_dataloader) # bs is 1
    visualization_dir = _visualization_output_dir(args, config)

    interval =  n_samples // 10
    source_dataset_name = config.dataset.train._base_.NAME 
    target_dataset_name = config.dataset.train.real_dataset
    with torch.no_grad():
        if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:
            for idx, data in enumerate(real_test_dataloader):

                target_gt,target_partial, _ = data 
                if args.use_gpu:
                    target_partial = target_partial.cuda()
                    target_gt = target_gt.cuda()

                rebuild_points = _model_rebuild_points(base_model(target_partial))

                rebuild_points = rebuild_points[-1]
                
                if idx <10:
                    visualize_point_cloud_batch(rebuild_points, f"pred_point_{idx}", visualization_dir)
                    visualize_point_cloud_batch(target_gt, f"gt_{idx}", visualization_dir)
                    visualize_point_cloud_batch(target_partial, f"partial_{idx}", visualization_dir)
                
                
                rebuild_loss_l1 =  ChamferDisL1(rebuild_points, target_gt)
                rebuild_loss_l2 =  ChamferDisL2(rebuild_points, target_gt)

                test_losses.update([ rebuild_loss_l1.item() * 10000, rebuild_loss_l2.item() * 10000])

                _metrics = Metrics.get(rebuild_points, target_gt)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)


                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()],
                        ['%.4f' % m for m in _metrics]), logger=logger)
        elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
            for idx, data in enumerate(real_test_dataloader):

                target_partial, _ = data   #[1,2048,3],[1,2048,3]
                if args.use_gpu:
                    target_partial = target_partial.float().cuda()
                   

                # Calculate losses for target
                rebuild_points = _model_rebuild_points(base_model(target_partial))#[1, 2048,3]
                rebuild_dense_points = rebuild_points[-1]
                uhd_rebuild_loss =  UHD_distance(target_partial.permute([0,2,1]),rebuild_dense_points.permute([0,2,1]))
                ucd_rebuild_loss =  UCD_distance(target_partial, rebuild_dense_points)

                test_losses.update([ucd_rebuild_loss.item()*10000, uhd_rebuild_loss.item()*100])

                _metrics = Metrics.get(target_partial,rebuild_dense_points)
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                test_metrics.update(_metrics)
            
            # ... (可视化部分的代码，如果不需要可以省略)

                if (idx + 1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, ['%.6f' % l for l in test_losses.val()],
                        ['%.6f' % m for m in _metrics]), logger=logger)

        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
    
    # Print testing results
    print_log('============================ TEST RESULTS ============================', logger=logger)
    print_log(_format_metric_header(test_metrics.items), logger=logger)
    print_log('Overall | ' + _format_metric_values(test_metrics.avg()), logger=logger)

    # Add testing results to TensorBoard
    """
    """
    if val_writer is not None:
        val_writer.add_scalar('DGPointMamba/Val/Rebuild_Loss_L1', test_losses.avg(0), epoch)
        val_writer.add_scalar('DGPointMamba/Val/Rebuild_Loss_L2', test_losses.avg(1), epoch)

        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('DGPointMamba/Val/Metric/%s' % metric, test_metrics.avg(i), epoch)

    if source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['3D_FUTURE', 'ModelNet']:

        return Metrics(config.consider_metric_2, test_metrics.avg())
    elif source_dataset_name == 'CRNShapeNet' and target_dataset_name in ['MatterPort', 'ScanNet','KITTI']:
        return Metrics(config.consider_metric_3, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    experiment_logger = ExperimentLogger(args, config)
    experiment_logger.write_run_meta(status='running')
    experiment_logger.write_summary(status='running')
    config.dataset.test._base_.SPLIT = 'test'
    _, test_dataloader = builder.real_dataset_builder(args,config.dataset.test)
    category_metrics = dict()
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    UnidirectionalCD = UnidirectionalChamferDistance()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionalCD, args, config, logger=logger, experiment_logger=experiment_logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, UnidirectionlCD ,args, config, logger = None, experiment_logger=None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['Rebuild_Loss_L1', 'Rebuild_Loss_L2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    visualization_dir = _visualization_output_dir(args, config)

    with torch.no_grad():
        for idx,data in enumerate(test_dataloader):
            
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test.real_dataset
            if dataset_name in ['3D_FUTURE','ModelNet']:
                partial = data[1].cuda()
                gt = data[0].cuda()

                rebuild_points = _model_rebuild_points(base_model(partial))

                dense_points = rebuild_points[-1]
                if idx < 500:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}", visualization_dir)
                    visualize_point_cloud_batch(gt, f"gt_{idx}", visualization_dir)
                    visualize_point_cloud_batch(partial, f"partial_{idx}", visualization_dir)

                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([dense_loss_l1.item() * 10000, dense_loss_l2.item() * 10000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                test_metrics.update(_metrics)
            elif dataset_name in ['KITTI','ScanNet','MatterPort']:
                partial = data[0].cuda()
                rebuild_points = _model_rebuild_points(base_model(partial))
                dense_points= rebuild_points[-1]
                if idx < 200:
                    visualize_point_cloud_batch(dense_points, f"pred_point_{idx}", visualization_dir)
                    visualize_point_cloud_batch(partial, f"partial_{idx}", visualization_dir)

                ucd_loss = UnidirectionlCD(partial, dense_points)

                _metrics = Metrics.get(partial, dense_points, require_emd=True)
                test_metrics.update(_metrics)


            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 100 == 0:
                print_log('Test[%d/%d]  Losses = %s Metrics = %s' %
                            (idx + 1, n_samples,  ['%.4f' % l for l in test_losses.val()], 
                            ['%.6f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            metrics_state = metric_dict(test_metrics.items, test_metrics.avg())
            _write_test_experiment_summary(experiment_logger, args, metrics_state)
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.6f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)

    print_log(_format_metric_header(test_metrics.items), logger=logger)
    print_log('Overall | ' + _format_metric_values(test_metrics.avg()), logger=logger)
    metrics_state = metric_dict(test_metrics.items, test_metrics.avg())
    _write_test_experiment_summary(experiment_logger, args, metrics_state)
    return 

def visualize_point_cloud_batch(batch_points, name, output_dir=None):


    if output_dir is None:
        output_dir = "point_visualization"
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(batch_points, torch.Tensor):
        batch_points = batch_points.detach().cpu().double().numpy()
    elif isinstance(batch_points, np.ndarray):
        batch_points = batch_points.astype(np.float64).copy()

    if batch_points.ndim == 3 and batch_points.shape[2] == 3:

        for i, points in enumerate(batch_points):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Must be 3D Coordinates")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 构建文件名并保存点云为 PLY 文件
            filename = os.path.join(output_dir, f"point_cloud_{i}_{name}.ply")
            o3d.io.write_point_cloud(filename, pcd)
