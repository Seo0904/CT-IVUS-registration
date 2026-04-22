import os
import numpy as np
from pydicom import sequence
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .sequence_ot import sequence_ot_loss_torch, get_valid_frame_idx

class WPSBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        print("始まるよー")
        """  Configures options specific for SB model
        """
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.05, help='weight for orthogonal regularization')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        # sequence OT の割当て情報を保存・可視化するためのデバッグ用オプション
        parser.add_argument('--save_ot_details', action='store_true', help='save sequence OT transport plan/details for debugging')
        parser.add_argument('--ot_details_max_samples', type=int, default=10, help='max number of OT detail samples to save')
        # sequence OT 用ハイパーパラメータ
        parser.add_argument('--seq_ot_iters', type=int, default=50, help='number of Sinkhorn iterations for sequence OT')
        parser.add_argument('--seq_ot_monotone', type=util.str2bool, nargs='?', const=True, default=True,
                    help='use monotone penalty in sequence OT')
        parser.add_argument('--seq_ot_monotone_penalty', type=float, default=50.0,
                    help='weight for monotone penalty in sequence OT')
        parser.add_argument('--use_identity_plan', action='store_true', help='use identity plan (matching frames in order) as additional supervision for sequence OT')
        parser.add_argument('--seq_ot_normalize', type=str, default='mean',
                    choices=['mean', 'median', 'max', 'none'],
                    help='normalization mode for OT cost matrix')
        # sequence OT のスナップショット保存頻度など
        parser.add_argument('--seq_ot_snapshot_epoch_interval', type=int, default=10,
                    help='epoch interval to save sequence OT snapshots (train.py)')
        parser.add_argument('--seq_ot_snapshot_num_samples', type=int, default=3,
                    help='number of sequences per split for OT snapshots (train.py)')
        parser.add_argument(
            '--sb_mode',
            type=str,
            default='original',
            choices=['none', 'original', 'seq_ot', 'both'],
            help='SB loss mode: original / seq_ot / both / none'
        )

        parser.add_argument(
            '--lambda_SB_original',
            type=float,
            default=1.0,
            help='weight for original SB loss'
        )

        parser.add_argument(
            '--lambda_SB_seq',
            type=float,
            default=1.0,
            help='weight for sequence OT loss'
        )
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']
        if self.isTrain and self.opt.sb_mode in ['seq_ot', 'both']:
            # SB_P: OT distance term (<P, M>), SB_U: monotone regularization term (weighted)
            self.loss_names += ['SB_P', 'SB_U']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real', 'real_B']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            if self.opt.sb_mode == "both":
                self.model_names = ['G', 'F', 'D', 'E']
            # E は OT ベースの SB では使わないので、WPSBModel では G/F/D のみ管理する
            else:
                self.model_names = ['G', 'F', 'D']

        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.sb_mode == "both":
                self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_E)

            # OT 詳細を保存する場合のカウンタと出力ディレクトリ
            if getattr(opt, 'save_ot_details', False):
                self.ot_details_saved = 0
                base_dir = os.path.join(opt.checkpoints_dir, opt.name, 'ot_details')
                os.makedirs(base_dir, exist_ok=True)
                self.ot_details_dir = base_dir

        # 勾配の流れをログに出すための簡易デバッグフラグ
        # 必要に応じて True/False を切り替えて使う想定
        self._grad_debug_enabled = True
        self._grad_debug_step = 0

    def _log_grad_stats_for_net(self, net, net_name, stage):
        """ネットワーク net について、勾配が付いているパラメータ数と
        全体の L2 ノルムを標準出力に出す。

        stage は "after_D_backward" など、呼び出し元の識別用文字列。
        """
        if (not self._grad_debug_enabled) or (net is None):
            return

        total_params = 0
        params_with_grad = 0
        sq_norm = 0.0

        for p in net.parameters():
            total_params += 1
            if p.grad is not None:
                params_with_grad += 1
                # 勾配ノルムを 2 乗しておいて最後に平方根を取る
                param_norm = p.grad.data.norm().item()
                sq_norm += param_norm * param_norm

        total_norm = sq_norm ** 0.5 if params_with_grad > 0 else 0.0
        print(f"[GRAD][{stage}] {net_name}: total_params={total_params}, with_grad={params_with_grad}, total_grad_norm={total_norm:.4e}")

    def _log_grad_stats(self, stage):
        """主要なネットワーク(G, D, F, E)について、勾配の有無と
        ノルムをまとめてログに出すヘルパー。
        """
        if not getattr(self, 'isTrain', False):
            return

        # 何回か呼び出したら自動で止めたい場合はここで制御する
        # 例: 最初の 100 ステップだけ表示
        max_steps = 100
        if self._grad_debug_step >= max_steps:
            if self._grad_debug_enabled:
                print(f"[GRAD][{stage}] reached max debug steps ({max_steps}); stop logging further grad stats.")
            self._grad_debug_enabled = False
            return

        self._grad_debug_step += 1

        # 各ネットワークごとにログ
        if hasattr(self, 'netD'):
            self._log_grad_stats_for_net(self.netD, 'netD', stage)
        if hasattr(self, 'netG'):
            self._log_grad_stats_for_net(self.netG, 'netG', stage)
        if hasattr(self, 'netF'):
            self._log_grad_stats_for_net(self.netF, 'netF', stage)
        if hasattr(self, 'netE'):
            self._log_grad_stats_for_net(self.netE, 'netE', stage)
            
    def data_dependent_initialize(self, data,data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        if hasattr(self, 'real_A2'):
            self.real_A2 = self.real_A2[:bs_per_gpu]
        if hasattr(self, 'real_B2'):
            self.real_B2 = self.real_B2[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()
            # 初期化時点でどのネットに勾配が流れているか確認
            self._log_grad_stats('data_dependent_after_G_backward')
            self.compute_D_loss().backward()
            self._log_grad_stats('data_dependent_after_D_backward')
            print(self.opt.sb_mode)
            print("sbモードはこれだよん")
            if self.opt.sb_mode == "both":
                self.compute_E_loss().backward()
                self._log_grad_stats('data_dependent_after_E_backward')
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netD.train()
        self.netF.train()
        if self.opt.sb_mode == 'both':
            self.netE.train()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        # Discriminator に勾配が流れているか確認
        self._log_grad_stats('optimize_after_D_backward')
        self.optimizer_D.step()

        if self.opt.sb_mode == 'both':
            # update E
            self.set_requires_grad(self.netE, True)
            self.optimizer_E.zero_grad()
            self.loss_E = self.compute_E_loss()
            self.loss_E.backward()
            # SB 用ネットワーク E への勾配を確認
            self._log_grad_stats('optimize_after_E_backward')
            self.optimizer_E.step()

        # update G
        self.set_requires_grad(self.netD, False)
        if self.opt.sb_mode == 'both':
            self.set_requires_grad(self.netE, False)
        
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        # Generator / Feature netF に勾配が流れているか確認
        self._log_grad_stats('optimize_after_G_backward')
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()       
        
    def set_input(self, input,input2=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
        else:
            # セカンドパスが無いときは、同じ入力をコピーして使う（ノイズだけ別になる想定）
            self.real_A2 = self.real_A.clone()
            self.real_B2 = self.real_B.clone()
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times]) #[0, 0.5, 0.74, 0.86, 0.94, 1]
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
        self.time_idx = time_idx ##何番目のstepか
        self.timestep     = times[time_idx] ##そのステップの時刻
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):
                
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                Xt = self.real_A if (t == 0) else (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)

                Xt_in, Xt_meta = self._flatten_seq_for_net(Xt)
                n_xt = Xt_in.shape[0]

                time_idx = torch.full((n_xt,), t, device=self.real_A.device, dtype=torch.long)
                z = torch.randn(size=[n_xt, 4 * self.opt.ngf], device=self.real_A.device)

                Xt_1 = self.netG(Xt_in, time_idx, z)
                Xt_1 = self._restore_seq_from_net(Xt_1, Xt_meta)
                
                Xt2 = self.real_A2 if (t == 0) else (1 - inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)

                Xt2_in, Xt2_meta = self._flatten_seq_for_net(Xt2)
                n_xt2 = Xt2_in.shape[0]

                time_idx = torch.full((n_xt2,), t, device=self.real_A.device, dtype=torch.long)
                z = torch.randn(size=[n_xt2, 4 * self.opt.ngf], device=self.real_A.device)

                Xt_12 = self.netG(Xt2_in, time_idx, z)
                Xt_12 = self._restore_seq_from_net(Xt_12, Xt2_meta)
                
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1 - inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)

                    XtB_in, XtB_meta = self._flatten_seq_for_net(XtB)
                    n_xtb = XtB_in.shape[0]

                    time_idx = torch.full((n_xtb,), t, device=self.real_A.device, dtype=torch.long)
                    z = torch.randn(size=[n_xtb, 4 * self.opt.ngf], device=self.real_A.device)

                    Xt_1B = self.netG(XtB_in, time_idx, z)
                    Xt_1B = self._restore_seq_from_net(Xt_1B, XtB_meta)
            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
                      
        
        z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
        z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                if self.real.dim() == 5:
                    self.real = torch.flip(self.real, [4])
                    self.realt = torch.flip(self.realt, [4])
                else:
                    self.real = torch.flip(self.real, [3])
                    self.realt = torch.flip(self.realt, [3])

        realt_in, realt_meta = self._flatten_seq_for_net(self.realt)
        n_realt = realt_in.shape[0]
        time_idx_realt = torch.full((n_realt,), int(self.time_idx[0]), device=self.real_A.device, dtype=torch.long)
        z_in = torch.randn(size=[n_realt, 4 * self.opt.ngf], device=self.real_A.device)

        self.fake = self.netG(realt_in, time_idx_realt, z_in)
        self.fake = self._restore_seq_from_net(self.fake, realt_meta)

        realA2_in, realA2_meta = self._flatten_seq_for_net(self.real_A_noisy2)
        n_realA2 = realA2_in.shape[0]
        time_idx_realA2 = torch.full((n_realA2,), int(self.time_idx[0]), device=self.real_A.device, dtype=torch.long)
        z_in2 = torch.randn(size=[n_realA2, 4 * self.opt.ngf], device=self.real_A.device)

        self.fake_B2 = self.netG(realA2_in, time_idx_realA2, z_in2)
        self.fake_B2 = self._restore_seq_from_net(self.fake_B2, realA2_meta)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        
        if self.opt.phase == 'test':
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    Xt_in, Xt_meta = self._flatten_seq_for_net(Xt)
                    n_xt = Xt_in.shape[0]

                    time_idx = torch.full((n_xt,), t, device=self.real_A.device, dtype=torch.long)
                    z = torch.randn(size=[n_xt, 4 * self.opt.ngf], device=self.real_A.device)

                    Xt_1 = self.netG(Xt_in, time_idx, z)
                    Xt_1 = self._restore_seq_from_net(Xt_1, Xt_meta)
                    
                    setattr(self, "fake_"+str(t+1), Xt_1)
                    
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)
        # fake_B は (B,T,C,H,W) の場合があるので、D には (B*T,C,H,W) を渡す
        fake = self.fake_B.detach()
        fake_use, _ = self._flatten_seq_for_net(fake)

        std = torch.rand(size=[1]).item() * self.opt.std
        pred_fake = self.netD(fake_use, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        # real_B のゼロ画素フレームを除外
        real_B = self.real_B
        real_B_flat, _ = self._flatten_seq_for_net(real_B)
        if real_B_flat.dim() == 4:
            valid_idx = get_valid_frame_idx(real_B_flat)
            if valid_idx.numel() > 0 and valid_idx.numel() < real_B_flat.shape[0]:
                real_B_flat = real_B_flat[valid_idx]
        real_B_use = real_B_flat

        self.pred_real = self.netD(real_B_use, self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=2)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=2)

        XtXt_1_flat, _ = self._flatten_seq_for_net(XtXt_1)
        XtXt_2_flat, _ = self._flatten_seq_for_net(XtXt_2)

        time_idx_E = torch.full(
            (XtXt_1_flat.shape[0],),
            int(self.time_idx[0]),
            device=XtXt_1_flat.device,
            dtype=torch.long
        )

        temp = torch.logsumexp(
            self.netE(XtXt_1_flat, time_idx_E, XtXt_2_flat).reshape(-1),
            dim=0
        )

        self.loss_E = (
            -self.netE(XtXt_1_flat, time_idx_E, XtXt_1_flat).mean()
            + temp
            + temp ** 2
        )
        return self.loss_E

    def compute_G_loss(self):
        bs =  self.real_A.size(0)
        tau = self.opt.tau
        
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std
        
        if self.opt.lambda_GAN > 0.0:
            # D には 4D テンソルを渡す
            fake_use, _ = self._flatten_seq_for_net(fake)
            pred_fake = self.netD(fake_use, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        self.loss_SB_P = 0.0
        self.loss_SB_U = 0.0
        if self.opt.lambda_SB > 0.0:
            #ver1 同様、各シーケンスごとに (T,C,H,W) で OT を計算する
            if self.opt.sb_mode == 'seq_ot':
                self.loss_SB_seq = self.compute_sequence_ot_loss()
                self.loss_SB = self.opt.lambda_SB_seq * self.loss_SB_seq

            elif self.opt.sb_mode == 'both':
                self.loss_SB_original = self.compute_original_sb_loss()
                self.loss_SB_seq = self.compute_sequence_ot_loss()
                self.loss_SB = (
                    self.opt.lambda_SB_original * self.loss_SB_original
                    + self.opt.lambda_SB_seq * self.loss_SB_seq
                )

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            # real_B/idt_B のゼロ画素フレームを除外して NCE を計算
            real_B = self.real_B
            idt_B = self.idt_B
            if real_B.dim() == 5:
                b, t, c, h, w = real_B.shape
                real_B_reshape = real_B.view(b * t, c, h, w)
                idt_B_reshape = idt_B.view(b * t, c, h, w)
                valid_idx = get_valid_frame_idx(real_B_reshape)
                if valid_idx.numel() > 0 and valid_idx.numel() < real_B_reshape.shape[0]:
                    real_B_reshape = real_B_reshape[valid_idx]
                    idt_B_reshape = idt_B_reshape[valid_idx]
                real_B_use = real_B_reshape
                idt_B_use = idt_B_reshape
            else:
                real_B_use = real_B
                idt_B_use = idt_B

            self.loss_NCE_Y = self.calculate_NCE_loss(real_B_use, idt_B_use)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both
        return self.loss_G


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        src_in, _ = self._flatten_seq_for_net(src)
        tgt_in, _ = self._flatten_seq_for_net(tgt)

        n = src_in.shape[0]
        z = torch.randn(size=[n, 4 * self.opt.ngf], device=self.real_A.device)
        time_idx = torch.zeros(n, device=self.real_A.device, dtype=torch.long)

        feat_q = self.netG(tgt_in, time_idx, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src_in, time_idx, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def _save_ot_details(self, details, real_A_seq=None, real_B_seq=None, fake_B_seq=None):
        """sequence_ot_loss_torch から返された OT 割当て情報と、
        対応する realA / realB / fakeB シーケンスをファイルに保存し、
        可能なら簡易なヒートマップ画像も出力する。
        """
        if not hasattr(self, 'ot_details_dir'):
            return

        idx = getattr(self, 'ot_details_saved', 0)
        base = os.path.join(self.ot_details_dir, f"ot_{idx:04d}")

        # テンソルを CPU numpy に変換
        P = details.get('P')
        M = details.get('M')
        U = details.get('U')
        valid_idx = details.get('valid_idx')
        ot_cost = details.get('ot_cost')
        reg_cost = details.get('reg_cost')
        mono_loss = details.get('mono_loss')
        total = details.get('total')

        # 追加で保存したいシーケンス（形状は (T, C, H, W) を想定）
        real_A_arr = None
        real_B_arr = None
        fake_B_arr = None
        try:
            if real_A_seq is not None:
                real_A_arr = real_A_seq.detach().cpu().numpy()
            if real_B_seq is not None:
                real_B_arr = real_B_seq.detach().cpu().numpy()
            if fake_B_seq is not None:
                fake_B_arr = fake_B_seq.detach().cpu().numpy()
        except Exception:
            # 変換に失敗しても OT 自体の保存は続行する
            pass

        np.savez(
            base + '.npz',
            P=P.detach().cpu().numpy() if P is not None else None,
            M=M.detach().cpu().numpy() if M is not None else None,
            U=U.detach().cpu().numpy() if U is not None else None,
            valid_idx=valid_idx.detach().cpu().numpy() if valid_idx is not None else None,
            ot_cost=float(ot_cost.detach().cpu().item()) if ot_cost is not None else None,
            reg_cost=float(reg_cost.detach().cpu().item()) if reg_cost is not None else None,
            mono_loss=float(mono_loss.detach().cpu().item()) if mono_loss is not None else None,
            total=float(total.detach().cpu().item()) if total is not None else None,
            real_A=real_A_arr,
            real_B=real_B_arr,
            fake_B=fake_B_arr,
        )

        # 数値をぱっと確認できるように txt でも保存
        try:
            with open(base + '.txt', 'w') as f:
                f.write('# sequence OT details\n')

                if ot_cost is not None:
                    f.write(f'ot_cost (distance): {float(ot_cost.detach().cpu().item()):.6f}\n')
                if reg_cost is not None:
                    f.write(f'reg_cost (monotone weighted): {float(reg_cost.detach().cpu().item()):.6f}\n')
                if mono_loss is not None:
                    f.write(f'mono_loss (raw): {float(mono_loss.detach().cpu().item()):.6f}\n')
                if total is not None:
                    f.write(f'total (distance + reg): {float(total.detach().cpu().item()):.6f}\n')
                if (ot_cost is not None) or (reg_cost is not None) or (mono_loss is not None) or (total is not None):
                    f.write('\n')

                if P is not None:
                    P_np = P.detach().cpu().numpy()
                    f.write(f'P shape: {P_np.shape}\n')
                    np.savetxt(f, P_np, fmt='%.6f')
                    f.write('\n')

                if M is not None:
                    M_np = M.detach().cpu().numpy()
                    f.write(f'M shape: {M_np.shape}\n')
                    # コスト行列は大きくなりがちなので、統計量も一緒に書いておく
                    f.write(f'M stats: min={M_np.min():.6f}, max={M_np.max():.6f}, mean={M_np.mean():.6f}\n')
                    np.savetxt(f, M_np, fmt='%.6f')
                    f.write('\n')

                if U is not None:
                    U_np = U.detach().cpu().numpy().reshape(-1)
                    f.write(f'U shape: {U_np.shape}\n')
                    np.savetxt(f, U_np[np.newaxis, :], fmt='%.6f')
                    f.write('\n')

                if valid_idx is not None:
                    vidx_np = valid_idx.detach().cpu().numpy().reshape(-1)
                    f.write(f'valid_idx shape: {vidx_np.shape}\n')
                    np.savetxt(f, vidx_np[np.newaxis, :], fmt='%d')
                    f.write('\n')

        except Exception:
            # txt 出力に失敗しても学習は続行する
            pass

        # 可能なら P のヒートマップと、対応する real_A / real_B / fake_B の画像を保存
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as _np

            if P is not None:
                P_cpu = P.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(P_cpu, aspect='auto', origin='lower')
                ax.set_xlabel('valid tgt index')
                ax.set_ylabel('fake frame index')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(base + '_P.png')
                plt.close(fig)

            if M is not None:
                M_cpu = M.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(M_cpu, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('valid tgt index')
                ax.set_ylabel('fake frame index')
                ax.set_title('cost matrix M')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(base + '_M.png')
                plt.close(fig)

            if U is not None:
                U_cpu = U.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(U_cpu)
                ax.set_xlabel('fake frame index')
                ax.set_ylabel('barycentric tgt idx')
                fig.tight_layout()
                fig.savefig(base + '_U.png')
                plt.close(fig)

            # real_A / real_B / fake_B のシーケンスを横に並べた画像として保存
            def _save_seq_image(seq_arr, path, title):
                """seq_arr: (T, C, H, W) or (T, H, W) を想定"""
                if seq_arr is None:
                    return

                arr = _np.array(seq_arr)
                if arr.ndim == 4:  # (T, C, H, W)
                    T, C, H, W = arr.shape
                elif arr.ndim == 3:  # (T, H, W)
                    T, H, W = arr.shape
                    C = 1
                    arr = arr[:, _np.newaxis, ...]
                else:
                    return

                # [-1,1] 想定なので [0,1] に正規化
                arr = (arr + 1.0) / 2.0
                arr = _np.clip(arr, 0.0, 1.0)

                # 横一列にフレームを並べたキャンバスを作成
                if C == 1:
                    canvas = _np.zeros((H, W * T), dtype=_np.float32)
                    for t in range(T):
                        canvas[:, t * W:(t + 1) * W] = arr[t, 0]
                    fig, ax = plt.subplots(figsize=(1.5 * T, 3))
                    ax.imshow(canvas, cmap='gray', vmin=0.0, vmax=1.0)
                else:
                    canvas = _np.zeros((H, W * T, C), dtype=_np.float32)
                    for t in range(T):
                        frame = arr[t].transpose(1, 2, 0)  # (H, W, C)
                        canvas[:, t * W:(t + 1) * W, :] = frame
                    fig, ax = plt.subplots(figsize=(1.5 * T, 3))
                    ax.imshow(canvas, vmin=0.0, vmax=1.0)

                ax.set_axis_off()
                ax.set_title(title)
                fig.tight_layout()
                fig.savefig(path)
                plt.close(fig)

            _save_seq_image(real_A_arr, base + '_real_A.png', 'real_A sequence')
            _save_seq_image(real_B_arr, base + '_real_B.png', 'real_B sequence')
            _save_seq_image(fake_B_arr, base + '_fake_B.png', 'fake_B sequence')
        except Exception:
            # matplotlib が無い / 画像保存に失敗した場合でも学習は続行する
            pass

    def _flatten_seq_for_net(self, x):
        """
        x:
        frame input  -> (B, C, H, W)
        seq input    -> (B, T, C, H, W)

        returns:
        x_flat, meta
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b * t, c, h, w), (b, t, c, h, w)
        elif x.dim() == 4:
            return x, None
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}, shape={x.shape}")


    def _restore_seq_from_net(self, x, meta):
        """
        x: (B*T, C, H, W) or (B, C, H, W)
        """
        if meta is None:
            return x
        b, t, c, h, w = meta
        return x.reshape(b, t, c, h, w)

    def compute_original_sb_loss(self):
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=2)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=2)

        XtXt_1_flat, meta1 = self._flatten_seq_for_net(XtXt_1)
        XtXt_2_flat, meta2 = self._flatten_seq_for_net(XtXt_2)
        time_idx_flat = torch.full(
            (XtXt_1_flat.shape[0],),
            int(self.time_idx[0]),
            device=XtXt_1_flat.device,
            dtype=torch.long
        )   

        ET_XY = (
            self.netE(XtXt_1_flat, time_idx_flat, XtXt_1_flat).mean()
            - torch.logsumexp(
                self.netE(XtXt_1_flat, time_idx_flat, XtXt_2_flat).reshape(-1), dim=0
            )
        )

        loss_sb = (
            -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps
            * self.opt.tau * ET_XY
        )
        loss_sb = loss_sb + self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
        return loss_sb
    
    def compute_sequence_ot_loss(self):
        fake_seq = self.fake_B
        real_seq = self.real_B

        # sequence OT がどのテンソルから構成されていて、
        # 勾配を流せる状態かどうかを確認するためのログ
        try:
            print(
                "[GRAD][seq_ot] fake_seq shape=", tuple(fake_seq.shape),
                "requires_grad=", fake_seq.requires_grad,
                "grad_fn=", type(fake_seq.grad_fn).__name__ if fake_seq.grad_fn is not None else None,
                "| real_seq shape=", tuple(real_seq.shape),
                "requires_grad=", real_seq.requires_grad,
                "grad_fn=", type(real_seq.grad_fn).__name__ if real_seq.grad_fn is not None else None,
            )
        except Exception:
            # 形状取得などで失敗しても学習は継続させる
            pass

        if fake_seq.dim() == 5 and real_seq.dim() == 5:
            b, t, c, h, w = fake_seq.shape
            seq_ot = 0.0
            seq_ot_dist = 0.0
            seq_ot_reg = 0.0

            for i in range(b):
                ot_val, terms = sequence_ot_loss_torch(
                    fake_seq[i],
                    real_seq[i],
                    reg=self.opt.lmda,
                    iters=getattr(self.opt, 'seq_ot_iters', 50),
                    monotone=getattr(self.opt, 'seq_ot_monotone', True),
                    monotone_penalty=getattr(self.opt, 'seq_ot_monotone_penalty', 1.0),
                    normalize=(None if getattr(self.opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(self.opt, 'seq_ot_normalize', 'mean')),
                    return_details=False,
                    use_identity_plan=getattr(self.opt, 'seq_ot_use_identity_plan', False),
                )

                seq_ot = seq_ot + ot_val
                if terms is not None:
                    seq_ot_dist = seq_ot_dist + terms['ot_cost']
                    seq_ot_reg = seq_ot_reg + terms['reg_cost']

            loss_seq = seq_ot / b
            self.loss_SB_P = seq_ot_dist / b
            self.loss_SB_U = seq_ot_reg / b
        else:
            ot_val, terms = sequence_ot_loss_torch(
                fake_seq,
                real_seq,
                reg=self.opt.lmda,
                iters=getattr(self.opt, 'seq_ot_iters', 50),
                monotone=getattr(self.opt, 'seq_ot_monotone', True),
                monotone_penalty=getattr(self.opt, 'seq_ot_monotone_penalty', 1.0),
                normalize=(None if getattr(self.opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(self.opt, 'seq_ot_normalize', 'mean')),
                return_details=False,
                use_identity_plan=getattr(self.opt, 'seq_ot_use_identity_plan', False),
            )


            loss_seq = ot_val
            if terms is not None:
                self.loss_SB_P = terms['ot_cost']
                self.loss_SB_U = terms['reg_cost']

        # sequence OT だけを単体で backward したときに、fake_seq に
        # 実際に勾配が流れているかを直接確認するデバッグ用コード。
        # autograd.grad は .backward と違いパラメータの .grad を汚さないので、
        # retain_graph=True を指定しておけば学習本体の backward には影響しない。
        if getattr(self, '_grad_debug_enabled', False):
            try:
                g_fake = torch.autograd.grad(
                    loss_seq,
                    fake_seq,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                if g_fake is None:
                    print("[GRAD][seq_ot_debug] grad w.r.t fake_seq is None (no dependency?)")
                else:
                    grad_norm = g_fake.norm().item()
                    grad_max = g_fake.abs().max().item()
                    print(
                        "[GRAD][seq_ot_debug] grad(fake_seq): norm=",
                        f"{grad_norm:.4e}",
                        "max=",
                        f"{grad_max:.4e}",
                        "shape=",
                        tuple(g_fake.shape),
                    )
            except Exception as e:
                print("[GRAD][seq_ot_debug] error while computing grad:", str(e))

        return loss_seq
