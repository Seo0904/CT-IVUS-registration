import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .sequence_ot import sequence_ot_loss_torch

class WpsbModel(BaseModel):
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
        self.visual_names = ['real_A_seq', 'fake_B_seq', 'real_B_seq']
        if self.opt.phase == 'test':
            self.visual_names = ['real', 'real_B']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B_seq']

        if self.isTrain:
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
            
    def data_dependent_initialize(self, data, data2):
        """
        netF の初期化用に最初の forward を回す。
        ここで注意: data["A"] は (B,T,H,W) なので bs_per_gpu は「シーケンス数」。
        set_input後の self.real_A は (B*T,C,H,W) なので、[:bs_per_gpu] はやっちゃダメ。
        """
        bs_seq_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)

        self.set_input(data, data2)

        # --- debug print ---
        print("[DDI] after set_input:")
        print("  real_A_seq:", tuple(self.real_A_seq.shape))
        print("  real_B_seq:", tuple(self.real_B_seq.shape))
        print("  real_A    :", tuple(self.real_A.shape))
        print("  real_B    :", tuple(self.real_B.shape))

        # --- slice by sequence batch (B) ---
        self.real_A_seq = self.real_A_seq[:bs_seq_per_gpu]
        self.real_B_seq = self.real_B_seq[:bs_seq_per_gpu]
        Bsz, T, C, H, W = self.real_A_seq.shape

        # re-flatten for GAN/NCE
        self.real_A = self.real_A_seq.reshape(Bsz * T, C, H, W)
        self.real_B = self.real_B_seq.reshape(Bsz * T, C, H, W)

        print("[DDI] after slicing sequences:")
        print("  Bsz,T,C,H,W:", (Bsz, T, C, H, W))
        print("  real_A    :", tuple(self.real_A.shape), "numel=", self.real_A.numel())
        print("  real_B    :", tuple(self.real_B.shape), "numel=", self.real_B.numel())

        self.forward()

        if self.opt.isTrain:
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.opt.lr,
                    betas=(self.opt.beta1, self.opt.beta2)
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netD.train()
        self.netF.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        
        
        # update G
        self.set_requires_grad(self.netD, False)
        
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()       
        
    def set_input(self, input, input2=None):
        AtoB = self.opt.direction == 'AtoB'

        A = input['A' if AtoB else 'B'].to(self.device)
        B = input['B' if AtoB else 'A'].to(self.device)

        # A,B が (B,T,H,W) なら C=1 を付ける
        if A.dim() == 4:  # (B,T,H,W)
            A = A.unsqueeze(2)  # -> (B,T,1,H,W)
            B = B.unsqueeze(2)

        # ここで A,B は (B,T,C,H,W) を想定
        self.real_A_seq = A
        self.real_B_seq = B

        Bsz, T, C, H, W = A.shape

        # GAN/NCE用にフレームを畳む
        self.real_A = A.reshape(Bsz*T, C, H, W)
        self.real_B = B.reshape(Bsz*T, C, H, W)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        Bframe = self.real_A.size(0)
        device = self.real_A.device
        self.flipped_for_equivariance = False

        # time_idx
        self.time_idx = torch.zeros(Bframe, dtype=torch.long, device=device)

        z = torch.randn(Bframe, 4 * self.opt.ngf, device=device)
        self.fake_B = self.netG(self.real_A, self.time_idx, z)
        if hasattr(self, "real_A_seq"):
            Bsz, T, C, H, W = self.real_A_seq.shape
            # 念のため整合チェック
            # assert self.fake_B.numel() == Bsz * T * C * H * W, \
            #     f"fake_B numel mismatch: {self.fake_B.numel()} vs {Bsz*T*C*H*W}"
            self.fake_B_seq = self.fake_B.reshape(Bsz, T, C, H, W)

        # if self.isTrain:
        #     print("[FWD] fake_B:", tuple(self.fake_B.shape), "numel=", self.fake_B.numel())
        #     assert self.fake_B.shape[0] == Bframe, "fake_B batch != real_A batch (B*T) になってない"

        if self.opt.nce_idt and self.isTrain:
            z2 = torch.randn(Bframe, 4 * self.opt.ngf, device=device)
            self.idt_B = self.netG(self.real_B, self.time_idx, z2)
            if hasattr(self, "real_B_seq"):
                self.idt_B_seq = self.idt_B.reshape(Bsz, T, C, H, W)
                    
    def compute_D_loss(self):
        fake = self.fake_B.detach()

        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD(self.real_B, self.time_idx)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        return self.loss_D

    
    def compute_G_loss(self):
        fake = self.fake_B

        # GAN
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # Sequence OT
        self.loss_SB = 0.0
        if self.opt.lambda_SB > 0.0:
            Bsz, T, C, H, W = self.real_A_seq.shape

            # debug / assert: fake_B は (B*T,C,H,W) のはず
            # print("[G] reshape check:")
            # print("  fake_B:", tuple(self.fake_B.shape), "numel=", self.fake_B.numel())
            # print("  expect:", (Bsz * T, C, H, W), "expect_numel=", Bsz*T*C*H*W)
            # assert self.fake_B.numel() == Bsz*T*C*H*W, "fake_B の要素数が B*T*C*H*W と一致してない（どこかでB*Tが崩れてる）"

            fake_seq = self.fake_B.reshape(Bsz, T, C, H, W)

            seq_ot = 0.0
            for b in range(Bsz):
                seq_ot = seq_ot + sequence_ot_loss_torch(
                    fake_seq[b], self.real_B_seq[b],
                    reg=0.2, iters=50,
                    monotone=True, monotone_penalty=50.0,
                )
            self.loss_SB = seq_ot / Bsz

        # NCE
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = 0.5 * (self.loss_NCE + self.loss_NCE_Y)
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.netG(src, self.time_idx*0,z,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
