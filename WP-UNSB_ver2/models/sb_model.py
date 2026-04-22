import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
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
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
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
            self.model_names = ['G', 'F', 'D','E']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            
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
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()  
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
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
        times = np.concatenate([np.zeros(1),times])
        # GPU/CPU どちらでも動くように real_A と同じデバイスに配置
        times = torch.tensor(times, dtype=torch.float32, device=self.real_A.device)
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1], device=self.real_A.device) * torch.ones(size=[1], device=self.real_A.device)).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):
                
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


                Xt2      = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                Xt2_in, Xt2_meta = self._flatten_seq_for_net(Xt2)
                n_xt2 = Xt2_in.shape[0]

                time_idx = torch.full((n_xt2,), t, device=self.real_A.device, dtype=torch.long)
                z = torch.randn(size=[n_xt2, 4 * self.opt.ngf], device=self.real_A.device)

                Xt_12 = self.netG(Xt2_in, time_idx, z)
                Xt_12 = self._restore_seq_from_net(Xt_12, Xt2_meta)
                
                
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
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
            times = torch.tensor(times, dtype=torch.float32, device=self.real_A.device)
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1], device=self.real_A.device) * torch.ones(size=[1], device=self.real_A.device)).long()
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
            fake_use, _ = self._flatten_seq_for_net(fake)
            pred_fake = self.netD(fake_use, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
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

            self.loss_SB  = (
                -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps
                * self.opt.tau * ET_XY
            )
            self.loss_SB = self.loss_SB + self.opt.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
        
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        
        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both
        return self.loss_G


    def calculate_NCE_loss(self, src, tgt):
        """NCE 用特徴抽出。

        moving_mnist_paired から 5 次元 (B, T, C, H, W) が渡ってきても
        wpsb_model と同様に flatten してから netG/netF に通す。
        """
        n_layers = len(self.nce_layers)

        src_in, _ = self._flatten_seq_for_net(src)
        tgt_in, _ = self._flatten_seq_for_net(tgt)

        n = src_in.size(0)
        z = torch.randn(size=[n, 4 * self.opt.ngf], device=self.real_A.device)
        time_idx = torch.zeros(n, device=self.real_A.device, dtype=torch.long)

        feat_q = self.netG(tgt_in, time_idx, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and getattr(self, 'flipped_for_equivariance', False):
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src_in, time_idx, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def _flatten_seq_for_net(self, x):
        """wpsb_model と同じ 4D/5D 対応の flatten ヘルパー。

        x:
          - frame input: (B, C, H, W)
          - seq  input: (B, T, C, H, W)
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.view(b * t, c, h, w), (b, t, c, h, w)
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