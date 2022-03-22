import os
import pandas as pd
import numpy as np
from DLVR import compute_local

def calc_score(net, x):    
    N, C, H, W = x.shape
    y_bon_ori = net(x)
    y_bon = (y_bon_ori / 2 + 0.5) * H - 0.5
    ceiling_z = guess_ceiling(y_bon, H, W)
    target_local_2d = compute_local(y_bon, H, W, ceiling_z[None])
    
    kernel_size = 15
    unfold = nn.Unfold(kernel_size=(1, kernel_size))
    windows = unfold(target_local_2d.reshape(N, 2, W, 2).permute(0, 1, 3, 2)).reshape(N, 2, 2, -1, kernel_size) #(N, C, XY, L, K)
    windows_mean = windows.median(dim=-1, keepdim=True).values
    windows_slope = windows / windows.norm(dim=2, keepdim=True)    
    manhattan = torch.abs((windows - windows_mean) * torch.flip(windows_slope, [2])).mean(dim=4).min(dim=2).values.mean()
    
    ceil = target_local_2d[:, :1024, :]
    floor = target_local_2d[:, 1024:, :]
    ceil_floor = F.mse_loss(ceil, floor)    
    
    return manhattan + ceil_floor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='Path to the dataset to be split')
    parser.add_argument('--stored_file', required=True,
                        help='Path to store the split result')
    parser.add_argument('--pth', action='store_true',
                        help='If given, conduct the active selection based on the provided weight.')                    
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    dataset = PanoCorBonDataset(
        root_dir=args.root_dir, return_path=True)
    
    select_num = [50, 100, 200, 400, 1650]
    if args.pth is not None: # Active Selection
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(HorizonNet, args.pth).to(device)
        net = net.to(device)

        fileids = []
        scores = []
        for jth in trange(len(dataset)):
            x, _, _, path = dataset[jth]
            fileid = os.path.basename(path)[:-4]
            x = x[None]
            x = x.to(device)
            with torch.no_grad():
                score = calc_score(net, x, criterion)
                fileids.append(fileid)
                scores.append(score.item())
        scores = np.array(scores)
        
    else: # Random Selection
        fileids = [os.path.basename(path)[:-4] for _, _, _, path in dataset]
        scores = np.random.random(len(fileids))

    sorted_scores = np.sort(scores)[::-1]
    ths = [sorted_scores[n-1] for n in select_num]
    seqs = [(scores >= th).astype(int) for th in ths]

    result = {'fileid': fileids}
    for n, seq in zip(select_num, seqs):
        result[str(n)] = seq

    df = pd.DataFrame(result)
    df.to_csv(args.stored_file, index=None)
        