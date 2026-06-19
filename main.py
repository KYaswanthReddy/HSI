import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import torch
import torch.nn.functional as F
import argparse
from datasets import get_dataset, HyperX
from utils_HSI import sample_gt, metrics, seed_worker, set_requires_grad, preprocess_dataset
import time
import os
from datetime import datetime
from model.discriminator import discriminator
from model.pmg import Generator, Dis
from con_losses import SupConLoss
from sam import SAM


parser = argparse.ArgumentParser(description='PyTorch PMGDG')
parser.add_argument('--save_path', type=str, default='results/')
parser.add_argument('--data_path', type=str, default='datasets/Pavia/')#Houston Pavia hyrank 
parser.add_argument('--source_name', type=str, default='paviaU',    #paviaU Houston13 Dioni 
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',    #paviaC Houston18 Loukia
                    help='the name of the test dir')
parser.add_argument('--sample_nums', type=int, default=10,    
                    help='sample_nums of training')
parser.add_argument('--only_preprocess', action='store_true',
                    help='Run preprocessing only and exit before training')
parser.add_argument('--layers_num', type=int, default=10,    
                    help='layers_num of g')
parser.add_argument('--dim1', type=int, default=128,    
                    help='dim1 of g')
parser.add_argument('--dim2', type=int, default=8,    
                    help='dim2 of g')
parser.add_argument('--g_bool', type=bool, default=True,    
                    help='g_bool')
parser.add_argument('--sam_bool', type=bool, default=True,    
                    help='sam_bool')

group_pretrain = parser.add_argument_group('preTrain')
group_pretrain.add_argument('--pre_epoch_per_step', type=int, default=200)
group_pretrain.add_argument('--pre_lr', type=float, default=0.001)
group_pretrain.add_argument('--lambda_1', type=float, default=0.01)
group_pretrain.add_argument('--lambda_2', type=float, default=0.01)

group_train = parser.add_argument_group('Training')
group_train.add_argument('--temp', type=float, default=0.07, help='temperature for contrastive loss function')
group_train.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)Houston:11;Pavia:7")
group_train.add_argument('--lr', type=float, default=1e-1,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int, default=512,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--max_epoch', type=int, default=800)
group_train.add_argument('--sam_rho', type=float, default=0.05)
group_train.add_argument('--test_stride', type=int, default=1,
                    help="Sliding window step stride during inference (default = 1)")
group_train.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
group_train.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
group_train.add_argument('--seed', type=int, default=333,
                    help='random seed ')
group_train.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
group_train.add_argument('--log_interval', type=int, default=40)


group_model = parser.add_argument_group('model')
group_model.add_argument('--pro_dim', type=int, default=128)
group_model.add_argument("--GIN", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--adv", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--noise", type=bool, default=True, help='noise z')
group_model.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
group_model.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
group_model.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
group_model.add_argument('--GIN_ch', type=int, default=24, help='channel of GIN')


group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")
args = parser.parse_args()
def evaluate_pre(gnet, dnet, val_loader, gpu):
    ps = []
    ys = []
    for i,(x, y) in enumerate(val_loader):
        y = y - 1
        with torch.no_grad():
            x = x.to(gpu)
            x_sd = gnet(x)
            x = torch.cat((x, x_sd), dim=0)
            y = torch.cat((y, y), dim=0)
            p = dnet(x)
            p = p.argmax(dim=1)
            ps.append(p.detach().cpu().numpy())
            ys.append(y.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    results = metrics(ps, ys, n_classes=ys.max() + 1)
    return acc, results
def evaluate(net, val_loader, gpu):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    # print(ys.size)
    results = metrics(ps, ys, n_classes=ys.max() + 1)
    return acc, results
def generate_layers(start, n, multiplier):
    sequence = [start]
    current = start
    while current < n:
        current = int(current * multiplier)  
        if current > n:  
            break
        sequence.append(current)
    if sequence[-1] != n:
        sequence.append(n)
    return sequence

def experiment(log_dir = ''):
    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0,
        'finished': False
    }
    device = args.gpu
    hyperparams = vars(args)
    print(hyperparams)

    s = ''
    for k, v in args.__dict__.items():
        s += '\t' + k + '\t' + str(v) + '\n'

    f = open(log_dir + '/settings.txt', 'w+')
    f.write(s)
    f.close()

    seed_worker(args.seed) 
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS_src, RGB_BANDS_src, palette_src = get_dataset(args.source_name,
                                                            args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS_tar, RGB_BANDS_tar, palette_tar = get_dataset(args.target_name,
                                                            args.data_path)

    IGNORED_LABELS = list(set(IGNORED_LABELS_src) | set(IGNORED_LABELS_tar))
    print('load dataset')
    print('→ normalization')
    preprocess_dataset(args.source_name, img_src, gt_src, args.patch_size,
                       args.training_sample_ratio, IGNORED_LABELS_src,
                       rgb_bands=RGB_BANDS_src, palette=palette_src,
                       visualize=args.only_preprocess)
    preprocess_dataset(args.target_name, img_tar, gt_tar, args.patch_size,
                       args.training_sample_ratio, IGNORED_LABELS_tar,
                       rgb_bands=RGB_BANDS_tar, palette=palette_tar,
                       visualize=args.only_preprocess)
    if args.only_preprocess:
        print('→ stop execution')
        print('Preprocessing only mode completed.')
        return 0, 0

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    
    # if tmp < 1:
    #     for i in range(args.re_ratio-1):
    #         img_src_con = np.concatenate((img_src_con,img_src))
    #         train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
           

    hyperparams_train = hyperparams.copy()
    hyperparams_train['flip_augmentation'] = True
    hyperparams_train['radiation_augmentation'] = True

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    
    class_indices = {i: [] for i in range(1, 1+len(set(train_dataset.labels)))}#create a dictionary to store the indices of each class in the training dataset, where the keys are the class labels and the values are lists of indices corresponding to those labels.

    for idx, label in enumerate(train_dataset.labels):#iterate through the training dataset and append the index of each sample to the corresponding list in the class_indices dictionary based on its label.
        class_indices[int(label)].append(idx)

  
    selected_indices = []
    samples_per_class = args.sample_nums  
    for label, indices in class_indices.items():
        selected_indices += random.sample(indices, samples_per_class)
   
    # print(selected_indices)
    subset = torch.utils.data.Subset(train_dataset, selected_indices)
    train_dataset = subset
    
    # for i in range(args.re_ratio-1):
    #     img_src_con = np.concatenate((img_src_con,img_src))
    #     train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)

    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    batch_size=hyperparams['batch_size'])           
    cls_criterion = torch.nn.CrossEntropyLoss()
    if args.layers_num < 1:
        layers  =generate_layers(3, N_BANDS, 1.5)
        layers_num = len(layers)
    else:
        layers_num = args.layers_num 
        layers = [int((N_BANDS)/layers_num)*(i+1) for i in range(layers_num-1)]+[N_BANDS]
    g1 = Generator(imdim=N_BANDS, patch_size = hyperparams['patch_size'],layers = layers, dim1 = args.dim1, dim2 = args.dim2, device=device).to(args.gpu)
    g2 = Generator(imdim=N_BANDS, patch_size = hyperparams['patch_size'],layers = layers, dim1 = args.dim1, dim2 = args.dim2, device=device).to(args.gpu)
    d1 = Dis(imdim=N_BANDS, patch_size = hyperparams['patch_size'],layers = layers, proj=args.pro_dim, num_classes=num_classes).to(args.gpu)
    d2 = Dis(imdim=N_BANDS, patch_size = hyperparams['patch_size'],layers = layers, proj=args.pro_dim, num_classes=num_classes).to(args.gpu)
    
    G1_opt = torch.optim.Adam(g1.parameters(), lr=args.pre_lr)
    G2_opt = torch.optim.Adam(g2.parameters(), lr=2*args.pre_lr)
    D1_opt = torch.optim.Adam(d1.parameters(), lr=args.pre_lr)
    D2_opt = torch.optim.Adam(d2.parameters(), lr=args.pre_lr)
    con_criterion = SupConLoss(device=args.gpu)
    
    best_acc1 = 0
    best_kappa1 = 0
    best_acc2 = 0
    best_kappa2 = 0
    best_g1 = None
    best_g2 = None
    best_d1 = None
    best_d2 = None
    pre_epoch = layers_num * args.pre_epoch_per_step
    for epoch in range(pre_epoch):
        t1 = time.time()
        g1.train()
        g2.train()
        current_step =  int(epoch/(pre_epoch/layers_num)) + 1
        # print(f'pre_step:{current_step}')
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            with torch.no_grad():  
                x_g1, x_down1 = g1(x, current_step)#geneerated vs compresses/downsampled
                x_g2, x_down2 = g2(x, current_step)
            x_tgt1, x_down_tgt1  = g1(x, current_step)
            x_tgt2, x_down_tgt2  = g2(x, current_step)  
            p_SD1, z_SD1 = d1(x_down1, current_step = current_step, mode='train')#downsampled classification and features
            p_ED1, z_ED1 = d1(x_g1, current_step = current_step, mode='train')#generated classification and features    
            p_SD2, z_SD2 = d2(x_down2, current_step = current_step, mode='train')
            p_ED2, z_ED2 = d2(x_g2, current_step = current_step, mode='train')

            p_src0_1, z_whole1 = d1(x, current_step = layers_num, mode='train')#original classification and features
            p_src0_2, z_whole2 = d2(x, current_step = layers_num, mode='train')
            p_src1, z_down1 = d1(x_down_tgt1, current_step = current_step, mode='train')#downsampled classification and features
            p_src2, z_down2 = d2(x_down_tgt2, current_step = current_step, mode='train')
            zwd1 = torch.cat([z_whole1.unsqueeze(1), z_down1.unsqueeze(1)], dim=1)
            zwd2 = torch.cat([z_whole2.unsqueeze(1), z_down2.unsqueeze(1)], dim=1)
            wd_con_loss1 = con_criterion(zwd1, y, adv=False)#l3 loss between original and downsampled features
            wd_con_loss2 = con_criterion(zwd2, y, adv=False)
            loss_wd =  wd_con_loss1 + wd_con_loss2
            loss_wd.backward() #lg update the discriminator to make the features of the downsampled images close to the features of the original images, which encourages the generator to preserve more information in the downsampled features.

            zsrc1 = torch.cat([z_SD1.unsqueeze(1), z_ED1.unsqueeze(1)], dim=1)#downsampled and generated features for contrastive loss interclass comparison
            zsrc2 = torch.cat([z_SD2.unsqueeze(1), z_ED2.unsqueeze(1)], dim=1)
            
            src_cls_loss1 = cls_criterion(p_SD1, y.long()) + cls_criterion(p_ED1, y.long())# cls classification loss for downsampled and generated features
            src_cls_loss2 = cls_criterion(p_SD2, y.long()) + cls_criterion(p_ED2, y.long())
            
            
            p_tgt1, z_tgt1 = d1(x_tgt1, current_step = current_step, mode='train')#generated image  classification and features for target domain
            p_tgt2, z_tgt2 = d2(x_tgt2, current_step = current_step, mode='train')
            
            tgt_cls_loss1 = cls_criterion(p_tgt1, y.long())  #classification loss for generated features
            tgt_cls_loss2 = cls_criterion(p_tgt2, y.long()) 
            
            zall1 = torch.cat([z_tgt1.unsqueeze(1), zsrc1], dim=1)
            zall2 = torch.cat([z_tgt2.unsqueeze(1), zsrc2], dim=1)
            
            con_loss1 = con_criterion(zall1, y, adv=False)#generated+downsampled #l1
            con_loss2 = con_criterion(zall2, y, adv=False)
            loss1_1 = src_cls_loss1 + args.lambda_1 * con_loss1 + tgt_cls_loss1
            loss1_2 = src_cls_loss2 + args.lambda_1 * con_loss2 + tgt_cls_loss2
            D1_opt.zero_grad()  
            loss1_1.backward(retain_graph=True)  
            D2_opt.zero_grad() 
            loss1_2.backward(retain_graph=True)
            zsrc_con1 = torch.cat([z_tgt1.unsqueeze(1), z_ED1.unsqueeze(1).detach()], dim=1)
            zsrc_con2 = torch.cat([z_tgt2.unsqueeze(1), z_ED2.unsqueeze(1).detach()], dim=1)
            
            con_loss_adv1 = 0
            con_loss_adv2 = 0
            
            idx_1 = np.random.randint(0, zsrc1.size(1))
            idx_2 = np.random.randint(0, zsrc2.size(1))
            for i, id in enumerate(y.unique()):
                mask = y == y.unique()[i]
                z_SD_i1, zsrc_i1 = z_SD1[mask], zsrc_con1[mask]#downsampled features and generated features of the same class for contrastive loss intraclass comparison
                y_i1 = torch.cat([torch.zeros(z_SD_i1.shape[0]), torch.ones(z_SD_i1.shape[0])]) #labels for contrastive loss, where 0 for downsampled features and 1 for generated features
                zall1 = torch.cat([z_SD_i1.unsqueeze(1).detach(), zsrc_i1[:, idx_1:idx_1 + 1]], dim=0)#concatenate downsampled features and one randomly selected generated feature for the same class to compute contrastive loss, where the downsampled features are treated as anchors and the generated feature is treated as positive sample. This encourages the generated features to be close to the downsampled features of the same class in the feature space.
                if y_i1.size()[0] > 2:
                    con_loss_adv1 += con_criterion(zall1, y_i1)
                z_SD_i2, zsrc_i2 = z_SD2[mask], zsrc_con2[mask]
                y_i2 = torch.cat([torch.zeros(z_SD_i2.shape[0]), torch.ones(z_SD_i2.shape[0])])  
                zall2 = torch.cat([z_SD_i2.unsqueeze(1).detach(), zsrc_i2[:, idx_2:idx_2 + 1]], dim=0)
                
                if y_i2.size()[0] > 2:
                    con_loss_adv2 += con_criterion(zall2, y_i2)
            con_loss_adv1 = con_loss_adv1 / y.unique().shape[0] #l2
            loss2_1 = tgt_cls_loss1 + args.lambda_2 * con_loss_adv1 #classification loss for generated features and contrastive loss between generated features and downsampled features of the same class, which encourages the generated features to be close to the downsampled features of the same class in the feature space.
            con_loss_adv2 = con_loss_adv2 / y.unique().shape[0] 
            loss2_2 = tgt_cls_loss2 + args.lambda_2 * con_loss_adv2
            print(f'pre_epoch:{epoch}, loss2_1: {loss2_1:.2f}  loss2_2:{loss2_2:.2f}')
            G1_opt.zero_grad()
            loss2_1.backward()
            D1_opt.step()
            G1_opt.step()
            G2_opt.zero_grad()
            loss2_2.backward()
            D2_opt.step()
            G2_opt.step()
            
        d1.eval()
        d2.eval()
        
        teacc1, res1 = evaluate_pre(g1, d1, train_loader, args.gpu)
        teacc2, res2 = evaluate_pre(g2, d2, train_loader, args.gpu)

        if best_acc1 < teacc1:
            best_acc1 = teacc1
            best_kappa1 = res1["Kappa"]
            best_g1 = g1.state_dict()
            best_d1 = d1.state_dict()
        if best_acc2 < teacc2:
            best_acc2 = teacc2
            best_kappa2 = res2["Kappa"]
            best_g2 = g2.state_dict()
            best_d2 = d2.state_dict()
        if(int((epoch+1)/args.pre_epoch_per_step) + 1 != current_step):
            g1.load_state_dict(best_g1)
            g2.load_state_dict(best_g2)
            d1.load_state_dict(best_d1)
            d2.load_state_dict(best_d2)
        t2 = time.time()
    g1.load_state_dict(best_g1)
    g2.load_state_dict(best_g2)
    d1.load_state_dict(best_d1)
    d2.load_state_dict(best_d2)
    torch.save({'g1': g1.state_dict()}, os.path.join(log_dir, f'best_g1.pth'))
    torch.save({'g2': g2.state_dict()}, os.path.join(log_dir, f'best_g2.pth'))

    D_net = discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes, patch_size=hyperparams['patch_size']).to(args.gpu)

    if args.sam_bool:
        D_opt = torch.optim.SGD
        D_sam = SAM(D_net.parameters(), D_opt,rho=args.sam_rho , lr=args.lr, momentum=0.9)
    else:
        D_opt = torch.optim.Adam(D_net.parameters(), lr=args.lr)
    best_acc = 0
    best_kappa = 0
    for epoch in range(1,args.max_epoch+1):
        t1 = time.time()    
        loss_list = []
        D_net.train()
        D_net.mode = 'train'
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            if args.g_bool:
                with torch.no_grad():
                    x1,_ = g1(x,layers_num)
                    x2,_ = g2(x,layers_num)
                    y1 = y
                    y2 = y
                x = torch.cat((x,x1,x2),dim=0)
                y = torch.cat((y,y1,y2),dim=0)
            if args.sam_bool:
                D_sam.zero_grad()
                predict1 = D_net(x.detach())#cls+proj
                loss = cls_criterion(predict1, y.long())#classification loss for the discriminator, which encourages the discriminator to correctly classify the samples into their respective classes.
                loss.backward()
                D_sam.first_step(zero_grad=True)
                predict1 = D_net(x.detach())
                loss = cls_criterion(predict1, y.long())
                loss.backward()
                D_sam.second_step(zero_grad=True)
                loss_list.append(loss.item())
            else:
                D_opt.zero_grad()
                predict1 = D_net(x.detach())
                loss = cls_criterion(predict1, y.long())
                loss.backward()
                D_opt.step()
                loss_list.append(loss.item())
        loss_mean = np.mean(loss_list, 0)
        
        
        D_net.eval()
        D_net.mode = 'test'
        taracc, results = evaluate(D_net, test_loader, args.gpu)
        if best_acc < taracc:
            best_acc = taracc
            best_kappa = results["Kappa"]
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pth'))
            train_res['best_epoch'] = epoch
            train_res['best_acc'] = '{:.2f}'.format(best_acc)
            train_res['Confusion_matrix'] = '{:}'.format(results['Confusion_matrix'])
            train_res['OA'] = '{:.2f}'.format(results['Accuracy'])
            train_res['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
            train_res['F1scores'] = '{:}'.format(results["F1_scores"])
            train_res['kappa'] = '{:.4f}'.format(results["Kappa"])
        t2 = time.time()
        if epoch % args.log_interval == 0 or epoch == args.max_epoch:
            print(f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, loss_mean {loss_mean:.4f}  /// Test {len(test_loader.dataset)}, best_acc {best_acc:.2f}')

    with open(log_dir + '/train_log.txt', 'w+') as f:
        for key, value in train_res.items():
            f.write(f"{key}: {value}\n")
    f.close()
    return best_acc, best_kappa
    
def work():
    repeat_time = 1
    seeds = [333,111,222,444,555,666,777,888,999,0]
    mean_acc = 0.0
    mean_kappa = 0.0
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))#current path of main.py
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    exp_name = '{}/{}'.format(args.save_path, args.source_name+'to'+args.target_name+'_'+time_str)#it create a folder named by source name, target name and time in save path
    for i in range(repeat_time):
        args.seed = seeds[i]
        timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_dir = os.path.join(BASE_DIR, exp_name, 'lr_' + str(args.lr) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' +timestamp)
        log_dir = log_dir.replace('\\', '/')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print(f'{i+1} experiment')
        acc, kappa = experiment(log_dir)
        if args.only_preprocess:
            return
        print(f'experiment {i+1}, oa: {acc}, kappa: {kappa}')
        mean_acc += acc
        mean_kappa += kappa
    print(vars(args))
    print(f'{repeat_time} times experiments over, mean acc = {mean_acc/repeat_time}, mean kappa = {mean_kappa/repeat_time}')
    os.rename(exp_name, exp_name+f'_mean_acc{str(mean_acc)[:6]}_mean_kappa{str(mean_kappa*100)[:6]}')


if __name__=='__main__': ##If current file directly executed, then run the work function
    work()










