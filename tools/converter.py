# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> converter
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   06/10/2021 10:42
=================================================='''
import os
import os.path as osp
import pyquaternion
import numpy as np
from scipy.spatial.transform import Rotation as sciR

def compute_error():
    pred_pose = [0.766302005732681, -0.6324010686712784, 0.10619334434698134, 0.03966229910001037, 726.9730802651802, -18.65707414856966, -352.79095130927595]
    gt_pose = [0.7692734053595, -0.6311420052371, 0.0758244505556, 0.0642561260102, 731.4968602167638, 41.4718728508861, -339.9875856482615]
    q_error: 4.49
    t_error: 61.64

    pred_pose = [0.6589912169870195, -0.516797312050644, -0.3697803368086694, -0.4023849111247379, -464.83495683701074, -133.12552438377503, 509.9247164809101]
    gt_pose = [0.6580992040897, -0.5186389805211, -0.3705381066093, -0.4007749455879, -463.9186575044444, -130.9859740567904, 511.8186377276826]
    q_error: 0.47
    t_error: 5.70

    pred_Rcw = sciR.from_quat(quat=[pred_pose[1], pred_pose[2], pred_pose[3], pred_pose[0]]).as_dcm()
    pred_tcw = np.array(pred_pose[-3:], float).reshape(3, 1)
    pred_Rwc = pred_Rcw.transpose()
    pred_twc = -pred_Rcw.transpose() @ pred_tcw

    gt_Rcw = sciR.from_quat(quat=[gt_pose[1], gt_pose[2], gt_pose[3], gt_pose[0]]).as_dcm()
    gt_tcw = np.array(gt_pose[-3:], float).reshape(3, 1)
    gt_Rwc = gt_Rcw.transpose()
    gt_twc = -gt_Rcw.transpose() @ gt_tcw

    t_error = (pred_twc - gt_twc) ** 2
    t_error = np.sqrt(np.sum(t_error))
    print(t_error)


def analyze():
    log_dir = '/scratch2/fx221/exp/shloc/robotcar/nv'
    log_fn = 'nvgtsunl10c5r30_max_feats-resnete2-tr-n4096-r1024-0005-maskNNMfh0.950cluir50iv50rs12_full.log'

    error_ths = ((0.25, 2), (0.5, 5), (5, 10))
    success = [0, 0, 0]
    n_total = 0
    with open(osp.join(log_dir, log_fn), 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if l.find('All') < 0:
                continue
            l = l.split(' ')
            q_error = float(l[-2].split(':')[-1])
            t_error = float(l[-1].split(':')[-1])

            n_total += 1
            for idx, th in enumerate(error_ths):
                if t_error < th[0] and q_error < th[1]:
                    success[idx] += 1

    for idx, th in enumerate(error_ths):
        print(th, success[idx] / 460)


if __name__ == '__main__':
    analyze()
    exit(0)

    compute_error()
    exit(0)

    save_fn = '/data/cornucopia/fx221/localization/RobotCar-Seasons/3D-models/query_poses_v2.txt'
    result = {}
    with open(save_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')
            result[l[0]] = l[1:]
    
    with open(
            '/data/cornucopia/fx221/localization/RobotCar-Seasons/3D-models/query_poses_v2_gt.txt', 'w') as f:
        for fn in result.keys():
            pose = result[fn]
            camera = fn.split('/')[1]
            name = fn.split('/')[2]
            text = camera + '/' + name
            for v in pose:
                text += (' ' + v)
            f.write(text + '\n')
    exit(0)
    '''
    overcast-summer/rear/1432293723546724.jpg 0.7624105817892092 -0.6182855418071406 0.12928342994564576 0.14049515404304697 430.28281486645153 -58.86965905283495 240.56264034311948
    sun/rear/1425997445908933.jpg 0.760477877366836 -0.6213619087376974 0.12430469521186345 0.14188417513508228 430.1986815180605 -53.469662354678775 243.8360554340137
    '''
    pose_file = '/scratch2/fx221/localization/RobotCar-Seasons/robotcar_v2_train.txt'
    save_fn = '/data/cornucopia/fx221/localization/RobotCar-Seasons/3D-models/query_poses_v2.txt'

    results = {}
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split(' ')
            name = l[0]
            if name not in ['overcast-summer/rear/1432293723546724.jpg', 'sun/rear/1425997445908933.jpg']:
                continue
            Twc = np.array([float(v) for v in l[1:]], float).reshape(4, 4)
            Rwc = Twc[0:3, 0:3]
            twc = Twc[0:3, 3]
            Rcw = Rwc.transpose()
            tcw = - Rcw @ twc
            # Tcw = np.linalg.inv(Twc)
            Rcw = sciR.from_matrix(Rcw)
            qvec = Rcw.as_quat()
            # qvec = [qvec[3], qvec[0], qvec[1], qvec[2]]
            tvec = tcw.reshape(-1, )

            results[name] = [qvec[3], qvec[0], qvec[1], qvec[2], tvec[0], tvec[1], tvec[2]]

            print(name)
            print(Twc)
            print([qvec[3], qvec[0], qvec[1], qvec[2], tvec[0], tvec[1], tvec[2]])

    # exit(0)

    with open(save_fn, 'w') as f:
        for fn in results.keys():
            pose = results[fn]
            text = '{:s} {:.13f} {:.13f} {:.13f} {:.13f} {:.13f} {:.13f} {:.13f}'.format(fn, pose[0], pose[1], pose[2],
                                                                                  pose[3], pose[4], pose[5], pose[6])
            f.write(text + '\n')
