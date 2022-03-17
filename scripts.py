import os
import itertools

if __name__ == '__main__':
    zillow = '--train_root_dir /mnt/home_6T/sunset/zind --valid_root_dir /mnt/home_6T/sunset/zind'
    mp3d = '--train_root_dir /mnt/home_6T/sunset/mp3d_layout/train_no_occ --valid_root_dir /mnt/home_6T/sunset/mp3d_layout/valid_no_occ'
    # mp3d = '--train_root_dir /mnt/home_6T/sunset/mp3d_layout/train --valid_root_dir /mnt/home_6T/sunset/mp3d_layout/valid'
    zillow_mp3d = '--train_root_dir /mnt/home_6T/sunset/zind --valid_root_dir /mnt/home_6T/sunset/mp3d_layout/valid_no_occ'
    # cmds = [
    #     f'python semi_train.py --id ssp_and_sp_0199 {dataset} --batch_size_train 2 --sup_ratio 0.01 --epochs 20 > logs/ssp_and_sp_0199.log',
    #     f'python semi_train.py --id ssp_and_sp_0298 {dataset} --batch_size_train 2 --sup_ratio 0.02 --epochs 20 > logs/ssp_and_sp_0298.log',
    #     f'python semi_train.py --id ssp_and_sp_0397 {dataset} --batch_size_train 2 --sup_ratio 0.03 --epochs 20 > logs/ssp_and_sp_0397.log'
    # ]

    # ids = ['semi_0595', 'semi_1090', 'semi_1585']
    # ckpts = ['best_valid_1.pth', 'best_valid_5.pth', 'best.pth']
    # it = itertools.product(ids, ckpts)

    # cmds = [f'python inference.py --pth ckpt/{Id}/{ckpt} --img_glob "/mnt/home_6T/public/joshua/zind/val/*/label_cor_noocc/*.txt" --output_dir val_out' for Id, ckpt in it]
    # Id = 'semi_0199'
    # ckpt = 'best_valid_1.pth'
    # cmds = [f'python inference.py --pth ckpt/{Id}/{ckpt} --img_glob "/mnt/home_6T/public/joshua/zind/val/*/label_cor_noocc/*.txt" --output_dir val_out']    

    # try:
    #     assert False
    # except AssertionError as e:
    #     print('Catch')
    #     assert False

    # exid = 'validmap'
    # exids = ['ssp_and_sp_0199', 'ssp_and_sp_0298', 'ssp_and_sp_0397']
    
    # cmds = [f'python semi_train.py --id {exids[exid]}_{i} {dataset} --batch_size_train 2 --sup_ratio {0.01 * (exid + 1)} --epochs 20 > logs/{exids[exid]}_{i}.log' for exid, i in itertools.product(range(3), range(5))]
    
    # cmds = [f'python semi_train.py --id smp_and_sp_0199 {dataset} --batch_size_train 2 --sup_ratio 0.01 --epochs 10 > logs/smp_and_sp_0199_{i}.log' for i in range(5)] \
    #     + [f'python semi_train.py --id sp_01 {dataset} --batch_size_train 2 --sup_ratio 0.01 --no_self --epochs 10 > logs/sp_01.log']

    # cmds = [f'python semi_pretrain.py --id double_{i} {dataset} --batch_size_train 2 --sup_ratio 0.001 --epochs 20 > logs/double_{i}.log' for i in range(5)]

    # cmds = [f'python sup_train.py --id mp3d_scratch {mp3d} --batch_size_train 4 --epochs 100 > logs/mp3d_scratch.log']
    # cmds = [f'python inference.py --pth ckpt/{Id}/{ckpt} --img_glob "/mnt/home_6T/public/joshua/zind/val/*/label_cor_noocc/*.txt" --output_dir val_out']

    num_mp3ds = [50, 100, 200, 400, 1650]
    # num_mp3ds = [50]
    # sup_ratios = [0.01, 0.02, 0.03, 0.05]
    
    run_file = 'semi_pretrain.py'
    # run_file = 'sup_train.py'
    # run_file = 'semi_train.py'

    # num_mp3ds = [50]
    # # idxs = list(range(1))
    # # it = itertools.product(num_mp3ds, idxs)
    # cmds = [f'python sup_train.py --id mp3d_finetune_{num}_{idx} {mp3d} --batch_size_train 2 --num_mp3d {num} --epochs 20 > logs/mp3d_finetune_{num}_{idx}.log' for num, idx in it]

    # pth = 'ckpt/test_on_mp3d_best/best.pth'
    # pth = 'ckpt/src_tgt_0/best_valid.pth'
    # cmds = [f'python sup_train.py --id zillow_finetune_{ratio} {zillow} --batch_size_train 4 --sup_ratio {ratio} --pth {pth} --epochs 30 > logs/zillow_finetune_{ratio}.log' for ratio in sup_ratios[-1:]] \
    #         + [f'python sup_train.py --id zillow_scratch_{ratio} {zillow} --batch_size_train 4 --sup_ratio {ratio} --epochs 20 > logs/zillow_scratch_{ratio}.log' for ratio in sup_ratios[1:-1]]



    # # # cmds = [f'python semi_pretrain.py --id test_on_mp3d_{i} {zillow_mp3d} --batch_size_train 2 --epochs 20 > logs/test_on_mp3d_{i}.log' for i in range(5)]
    # cmds_finetune = [f'python sup_train.py --id mp3d_finetune_new_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --pth {pth} --epochs 100 > logs/mp3d_finetune_new_{num}.log' for num in num_mp3ds]
    # cmds_scratch = [f'python sup_train.py --id mp3d_scratch_active_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --epochs 100 > logs/mp3d_scratch_active_{num}.log' for num in num_mp3ds]
    # # cmds_finetune = [f'python sup_train.py --id mp3d_finetune_ss360split_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --pth {pth} --epochs 100' for num in num_mp3ds]
    # # cmds_scratch = [f'python sup_train.py --id mp3d_scratch_ss360split_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --epochs 100' for num in num_mp3ds]

    # cmds_finetune = [f'python {run_file} --id mp3d_finetune_active_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --pth {pth} --epochs 100 > logs/mp3d_finetune_active_{num}.log' for num in num_mp3ds]
    # cmds_scratch = [f'python {run_file} --id mp3d_scratch_active_{num} {mp3d} --batch_size_train 4 --num_mp3d {num} --epochs 100 > logs/mp3d_scratch_active_{num}.log' for num in num_mp3ds]
    # cmds = cmds_scratch + cmds_finetune
    # cmds = cmds_finetune
    # cmds = cmds_scratch

    # cmds = [f'python semi_pretrain.py --id ssp_and_sp_0496_{i} {zillow} --batch_size_train 2 --sup_ratio 0.04 --epochs 20 > logs/ssp_and_sp_0496_{i}.log' for i in range(5)]

    # cur_id = 'src_tgt'
    # cur_id = 'geometric'
    # num_trial = 10
    # cmds = [f'python semi_pretrain.py --id {cur_id}_{i} {zillow_mp3d} --batch_size_train 1 --no_sup --epochs 20 > logs/{cur_id}_{i}.log' for i in range(1, num_trial)]


    # exids = ['mp3d_finetune_ss360split_100', 'mp3d_scratch_ss360split_100']
    exids = ['mp3d_finetune_ss360split_50', 'mp3d_scratch_ss360split_50', 'mp3d_finetune_ss360split_200', 'mp3d_scratch_ss360split_200', 'mp3d_finetune_ss360split_400', 'mp3d_scratch_ss360split_400', 'mp3d_finetune_ss360split_1650', 'mp3d_scratch_ss360split_1650']
    cmds = [f'python sup_train.py --id eval_{cur_id}_all {mp3d} --batch_size_train 1 --no_train --epochs 1 --pth ckpt/{cur_id}/best_valid.pth > eval_logs/eval_{cur_id}_all.log' for cur_id in exids ]

    # ckptids = [f'mp3d_finetune_ss360split_{num}' for num in num_mp3ds] + [f'mp3d_scratch_ss360split_{num}' for num in num_mp3ds]
    # ckptids = ['']
    # cmds = [f'python {run_file} --id {cur_id}_{i} {zillow} --batch_size_train 4 --epochs 20 --no_train > eval_logs/eval_{ckptid}.log' for ckptid in ckptids]
            
    # cmds = [f'python {run_file} --id {cur_id}_{i} {zillow} --batch_size_train 2 --epochs 20 --no_sup > logs/{cur_id}_{i}.log' for i in range(10)]



    for cmd in cmds:
        print('*' * 50)
        print(cmd)
        print()
        # print('*' * 50)

        os.system(cmd)