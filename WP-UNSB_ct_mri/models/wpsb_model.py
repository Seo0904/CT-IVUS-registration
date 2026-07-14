import os
import numpy as np
from pydicom import sequence
import torch
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .sequence_ot import sequence_ot_loss_torch, sequence_ot_loss, get_valid_frame_idx

class WPSBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        print("始まるよー")
        """  Configures options specific for SB model
        """
        parser.add_argument('--mode', type=str, default="sb", choices=['FastCUT', 'fastcut', 'sb'])

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
        parser.add_argument('--sinkhorn_type', type=str, default="sinkhorn", help='type of Sinkhorn algorithm for sequence OT: "sinkhorn" or "log"')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        # sequence OT の割当て情報を保存・可視化するためのデバッグ用オプション
        parser.add_argument('--save_ot_details', action='store_true', help='save sequence OT transport plan/details for debugging')
        parser.add_argument('--ot_details_max_samples', type=int, default=10, help='max number of OT detail samples to save')
        # netD 特徴空間（距離を測っている空間）の t-SNE 可視化。ot_details と同じタイミングで出す。
        parser.add_argument('--save_feature_tsne', action='store_true',
                            help='save t-SNE of netD penultimate feature space at ot-details timing '
                                 '(shape=modality CT/MR, color=pair/case)')
        parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                            help='t-SNE perplexity (auto-capped to (N-1)/3 for small N)')
        # sequence OT 用ハイパーパラメータ
        parser.add_argument('--seq_ot_iters', type=int, default=50, help='number of Sinkhorn iterations for sequence OT')
        parser.add_argument('--seq_ot_monotone', type=util.str2bool, nargs='?', const=True, default=True,
                    help='use monotone penalty in sequence OT')
        parser.add_argument('--seq_ot_monotone_penalty', type=float, default=50.0,
                    help='weight for monotone penalty in sequence OT')
        parser.add_argument('--seq_ot_p_entropy', type=util.str2bool, nargs='?', const=True, default=True,
                help='use row-wise entropy regularization term for transport plan P in sequence OT')
        parser.add_argument('--seq_ot_p_entropy_penalty', type=float, default=1.0,
                help='weight for P entropy regularization in sequence OT')
        parser.add_argument('--seq_ot_divergence', type=util.str2bool, nargs='?', const=True, default=False,
            help='use divergence regularization term based on fake-fake transport')
        parser.add_argument('--seq_ot_divergence_penalty', type=float, default=-0.5,
            help='weight for divergence regularization in sequence OT')
        parser.add_argument('--seq_ot_normalize', type=str, default='mean',
                    choices=['mean', 'median', 'max', 'none'],
                    help='normalization mode for OT cost matrix')
        # sequence OT のソルバ選択 (POT / GeomLoss) と geo 専用ハイパラ
        parser.add_argument('--seq_ot_solver', type=str, default='pot',
                    choices=['pot', 'geo'],
                    help='solver for sequence OT: "pot" (ot.sinkhorn) or "geo" (GeomLoss)')
        parser.add_argument('--seq_ot_geo_scaling', type=float, default=0.99,
                    help='[geo] epsilon-scaling ratio for GeomLoss Sinkhorn (higher=more accurate/slower)')
        parser.add_argument('--seq_ot_geo_p', type=int, default=2,
                    help='[geo] ground cost order p for GeomLoss (eps=blur**p, blur=reg**(1/p))')
        parser.add_argument('--seq_ot_unbalanced', type=float, default=None,
                    help='[geo] unbalanced OT marginal KL penalty rho. None=balanced (default). '
                         'Pass a positive float (e.g. 60) to relax marginals and drop unmatched frames. '
                         'rho は raw コストスケール基準なので seq_ot_normalize=none 前提で調整すること')
        parser.add_argument('--seq_ot_debias', type=util.str2bool, nargs='?', const=True, default=False,
                    help='[geo] use solve_sample(debias=True) Sinkhorn divergence as ot_cost '
                         '(gen->tgt と gen->gen を1回の計算で分離ログ; 手作り seq_ot_divergence は不要になる). '
                         'コストは sqeuclidean 固定で seq_ot_normalize は無視される')
        # sequence OT のコスト空間: 画像ピクセル / Discriminator 特徴空間
        parser.add_argument('--seq_ot_cost_space', type=str, default='pixel',
                    choices=['pixel', 'flatten', 'random_patch'],
                    help='space for sequence-OT frame cost. '
                         'pixel=image L2 (current). '
                         'flatten=netD penultimate feature flattened to one vector per frame. '
                         'random_patch=one random spatial location of netD feature per step '
                         '(unbiased lightweight approx of flatten).')
        # ===== OT-GAN (OT-GAN-main) 由来の学習法オプション =====
        parser.add_argument('--n_gen', type=int, default=0,
                    help='OT-GAN流のG/D排他交互更新。0=無効(毎step G/D両方更新=従来)。'
                         'N>0 で (step+1)%%N==0 の step は D のみ、それ以外は G のみ更新。'
                         '更新比率は G:D=(N-1):1（N=3で2:1、論文の3:1にするならN=4）。'
                         '参考コードの if i + 1 %% n_gen == 0 は優先順位バグ(Dが一度も更新'
                         'されない)なので、意図どおり (i+1) %% n_gen == 0 で実装している')
        parser.add_argument('--lr_D', type=float, default=None,
                    help='Dの学習率。未指定(None)=--lrと同じ(従来)。OT-GAN準拠はG/Dとも3e-4')
        parser.add_argument('--D_loss_mode', type=str, default='gan', choices=['gan', 'ot'],
                    help='Dのloss。gan=real/fake判定のpatchGAN(従来)。'
                         'ot=D特徴空間のseq-OT Sinkhorn divergenceを最大化'
                         '(OT-GANのminibatch energy distance critic相当。'
                         'divergence = OT(f,r) - 0.5*OT(f,f) - 0.5*OT(r,r) なので'
                         'クロス項-自己項の構造がOT-GANのcritic目的と同等)。'
                         'ot は seq_ot_cost_space=flatten/random_patch が必須で、'
                         'seq_ot_feat_norm=l2 を強く推奨(発散防止)')
        parser.add_argument('--seq_ot_metric', type=str, default='l2', choices=['l2', 'cosine'],
                    help='seq-OTコスト行列の距離。l2=cdist^p(従来)。'
                         'cosine=1 - x̂ŷᵀ(OT-GAN準拠、値域[0,2]、内部でL2正規化)。'
                         'geo debias(solve_sample)経路はsqeuclidean固定のため、cosine指定時は'
                         '特徴を単位球面化して計算(½‖x−y‖²=1−x·y なので係数2倍を除き等価)')
        parser.add_argument('--seq_ot_feat_norm', type=str, default='none', choices=['none', 'l2'],
                    help='コスト計算前にフレーム記述子をL2正規化(単位球面化)する。'
                         'コストスケールがO(1)に固定され reg(lmda) がコスト空間に依らず安定。'
                         'D_loss_mode=ot ではDが特徴ノルムを膨らませてOT距離を稼ぐ'
                         '自明解を防ぐため l2 を強く推奨')
        # D の更新を lambda_GAN から切り離して制御する。
        # auto: lambda_GAN>0 もしくは D 特徴 OT を使うとき D を学習（従来挙動を内包）
        parser.add_argument('--update_D', type=str, default='auto',
                    choices=['auto', 'always', 'never'],
                    help='whether to train the discriminator. auto=train if lambda_GAN>0 or '
                         'seq_ot_cost_space uses netD features; always/never to force.')
        # sequence OT のスナップショット保存頻度など
        parser.add_argument('--seq_ot_snapshot_epoch_interval', type=int, default=10,
                    help='epoch interval to save sequence OT snapshots (train.py)')
        parser.add_argument('--seq_ot_snapshot_num_samples', type=int, default=3,
                    help='number of sequences per split for OT snapshots (train.py)')
        parser.add_argument('--netg_chunk', type=int, default=32,
                    help='netG forward を frame 軸 (flatten した B*T 次元) で '
                         'この枚数ずつチャンク実行する。0=一括。'
                         'volume を 1 sequence として流す際、~200 枚を一括で '
                         '2D CNN に通すと中間テンソルが 32bit index 上限 (2^31) を '
                         '超えて RuntimeError になるため、その回避用。'
                         'frame を間引かないので seq-OT の対応は保持される。')
        parser.add_argument('--netg_checkpoint', type=util.str2bool, nargs='?',
                    const=True, default=True,
                    help='学習時、チャンクごとに gradient checkpointing を使い '
                         'activation を保存せず backward で再計算する。'
                         '全 frame を保持したまま backprop する際の OOM 回避用 '
                         '(計算は増えるがメモリは1チャンク分に抑えられる)。')
        parser.add_argument('--amp_netg', type=util.str2bool, nargs='?',
                    const=True, default=False,
                    help='netG forward のみ AMP(混合精度fp16)で実行する。'
                         '出力は fp32 に戻すので下流の seq-OT / D / E は fp32 のまま。'
                         'netG backprop の fp16 勾配アンダーフローは GradScaler で保護。'
                         'netG が律速のため高速化・省メモリ効果が大きい。')
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
        # UNSB の SB ドリフト項 (tau·‖Xt - fake_B‖²) を独立に加える重み。
        # これは「ただの MSE」ではなく Schrödinger Bridge 損失の第2項(SDE 離散化の
        # 遷移コスト)で、係数 tau を保持する。sb_mode='both' では compute_original_sb_loss
        # 内に同じ項が既にあるため、both で lambda_MSE>0 にすると二重加算になる(警告を出す)。
        # 既定 0.0 で従来挙動(後方互換)。seq_ot モードに UNSB 同様のドリフト項を入れたいとき用。
        parser.add_argument('--lambda_MSE', type=float, default=0.0,
                    help='weight for the UNSB SB drift term. loss_G += lambda_MSE*tau*mean(||Xt-fake_B||^2). '
                         'lambda_MSE=1 gives the same contribution as the term inside '
                         'compute_original_sb_loss (both). The logged "MSE" is the raw '
                         'mean((Xt-fake)^2) (without tau). 0=off(default). Note: sb_mode=both '
                         'already contains this term, so lambda_MSE>0 with both double-counts it.')
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

        # --- D 学習の派生フラグ(loss_names の登録判定に使うので先に決める)---
        self.d_loss_mode = getattr(self.opt, 'D_loss_mode', 'gan')
        # seq-OT を D 特徴空間で計算するか(=D を loss に使うか)の派生フラグ
        self.use_D_feat_ot = getattr(opt, 'seq_ot_cost_space', 'pixel') in ('flatten', 'random_patch')
        if self.isTrain and self.d_loss_mode == 'ot':
            if not self.use_D_feat_ot:
                raise ValueError(
                    "--D_loss_mode ot は D 特徴空間の OT を D の loss にするため、"
                    "--seq_ot_cost_space flatten か random_patch が必須です "
                    f"(現在: {getattr(opt, 'seq_ot_cost_space', 'pixel')})。"
                    "pixel では D のパラメータが loss に入らず勾配が流れません。"
                )
            if getattr(opt, 'seq_ot_feat_norm', 'none') == 'none':
                print(
                    "[warn] D_loss_mode=ot かつ seq_ot_feat_norm=none: D が特徴ノルムを"
                    "膨らませるだけで OT 距離を無限に稼げる自明解があり発散しやすい。"
                    "OT-GAN 同様 --seq_ot_feat_norm l2 (単位球面化) を強く推奨します。"
                )
        # D を学習するか。auto=lambda_GAN>0 もしくは D 特徴 OT を使うとき学習。
        _update_D = getattr(opt, 'update_D', 'auto')
        if _update_D == 'always':
            self.should_train_D = True
        elif _update_D == 'never':
            self.should_train_D = False
        else:  # auto
            self.should_train_D = (opt.lambda_GAN > 0.0) or self.use_D_feat_ot

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['SB']
        # UNSB ドリフト MSE 項(lambda_MSE>0 のとき loss にもログにも載せる)
        if self.isTrain and getattr(self.opt, 'lambda_MSE', 0.0) > 0.0:
            self.loss_names += ['MSE']
            if self.opt.sb_mode == 'both':
                print('[warn] lambda_MSE>0 かつ sb_mode=both: '
                      'compute_original_sb_loss 内の tau*||Xt-fake||^2 と二重加算になります。'
                      'seq_ot/none で使うか、both では lambda_MSE=0 を推奨。')
        if self.isTrain and self.opt.sb_mode in ['seq_ot', 'both']:
            # SB_P: OT distance, SB_U: monotone regularization, SB_ENT: entropy regularization, SB_DIV: divergence regularization
            self.loss_names += ['SB_P', 'SB_U', 'SB_ENT', 'SB_DIV']
            # 診断ログ: M_SUM(コスト行列の総和), M_DIAG(対角=単位行列1の位置の和=trace), P_IDENT_ERR(P と I の誤差)
            self.loss_names += ['M_SUM', 'M_DIAG', 'P_IDENT_ERR']
            # debias 時のみ: SB_GT (gen->tgt biased), SB_GG (gen->gen self) を別々にログ
            if getattr(self.opt, 'seq_ot_debias', False):
                self.loss_names += ['SB_GT', 'SB_GG']
            # gradient of OT loss w.r.t. fake_seq
            self.loss_names += ['SB_GRAD_NORM', 'SB_GRAD_MIN', 'SB_GRAD_MAX']
            # gradient of monotone(U) term only w.r.t. fake_seq（U からの勾配だけを切り出してログ）
            self.loss_names += ['SB_U_GRAD_NORM', 'SB_U_GRAD_MIN', 'SB_U_GRAD_MAX']
            # gradient of total G loss w.r.t. netG parameters (after backward)
            self.loss_names += ['NETG_GRAD_NORM', 'NETG_GRAD_MIN', 'NETG_GRAD_MAX']
            # netG パラメータへの勾配を GAN由来 / SB由来 に分離してログ（grad_stats_freq 間引き）
            self.loss_names += ['NETG_GRAD_GAN_NORM', 'NETG_GRAD_SB_NORM']
        if self.isTrain and self.opt.lambda_GAN > 0.0:
            # adversarial loss:
            #   G_GAN  : G が D を騙そうとするロス（lambda_GAN>0 のときだけ G に流れる）
            self.loss_names += ['G_GAN']
        # D が実際に学習されるとき(should_train_D)は lambda_GAN=0 でも D ログを出す。
        # D特徴OT(cost_space=flatten/random_patch)で D を学習するケースを含む。
        if self.isTrain and self.should_train_D:
            #   D      : Discriminator 全体ロス
            #            gan: (D_real + D_fake)/2 / ot: -SinkhornDiv(feat_fake, feat_real)
            #   D_real : real_B を本物と見抜くロス (gan モードのみ)
            #   D_fake : fake_B を偽物と見抜くロス (gan モードのみ)
            self.loss_names += ['D'] if self.d_loss_mode == 'ot' else ['D', 'D_real', 'D_fake']
            # gradient of D loss w.r.t. netD parameters (after backward) = D ロスのみ由来
            self.loss_names += ['NETD_GRAD_NORM', 'NETD_GRAD_MIN', 'NETD_GRAD_MAX']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real', 'real_B']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
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
            self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            # netG だけ AMP するときの GradScaler (G の backward/step 用)。
            # SB 損失が大きく既定の init_scale(65536) では序盤に fp16 overflow して
            # 数 step スキップされるため、安定値付近(2^13)から開始して warmup を省く。
            self.amp_netg = bool(getattr(opt, 'amp_netg', False))
            self.scaler = GradScaler(init_scale=2.0 ** 13, enabled=self.amp_netg)
            # adversarial loss を使うとき、もしくは D 特徴空間 OT で D を学習するとき optimizer を作る
            if self.should_train_D:
                # D の学習率は --lr_D で個別指定可（未指定なら G と同じ opt.lr）
                _lr_D = opt.lr_D if getattr(opt, 'lr_D', None) is not None else opt.lr
                self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=_lr_D, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)
            if self.opt.sb_mode == "both":
                self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                self.optimizer_E = torch.optim.AdamW(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_E)

            # OT 詳細を保存する場合のカウンタと出力ディレクトリ
            if getattr(opt, 'save_ot_details', False):
                self.ot_details_saved = 0
                base_dir = os.path.join(opt.checkpoints_dir, opt.name, 'ot_details')
                os.makedirs(base_dir, exist_ok=True)
                self.ot_details_dir = base_dir

        self._grad_debug_enabled = getattr(opt, 'debug', False)
        self._grad_debug_step = 0
        self._grad_log_iter = 0
        self._grad_stats_step = 0
        self.current_epoch = 0
        self.loss_SB_GRAD_NORM = 0.0
        self.loss_SB_GRAD_MIN = 0.0
        self.loss_SB_GRAD_MAX = 0.0
        self.loss_SB_U_GRAD_NORM = 0.0
        self.loss_SB_U_GRAD_MIN = 0.0
        self.loss_SB_U_GRAD_MAX = 0.0
        self.loss_NETG_GRAD_NORM = 0.0
        self.loss_NETG_GRAD_GAN_NORM = 0.0
        self.loss_NETG_GRAD_SB_NORM = 0.0
        self.loss_NETG_GRAD_MIN = 0.0
        self.loss_NETG_GRAD_MAX = 0.0
        # UNSB ドリフト MSE 項(ログ取得が先に来ても落ちないよう初期化)
        self.loss_MSE = 0.0
        # adversarial losses: 毎ステップ更新されるが、ログ取得が先に来ても落ちないよう初期化
        self.loss_G_GAN = 0.0
        self.loss_D = 0.0
        self.loss_D_real = 0.0
        self.loss_D_fake = 0.0
        self.loss_NETD_GRAD_NORM = 0.0
        self.loss_NETD_GRAD_MIN = 0.0
        self.loss_NETD_GRAD_MAX = 0.0

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
            # self.compute_D_loss().backward()
            self._log_grad_stats('data_dependent_after_D_backward')
            print(self.opt.sb_mode)
            print("sbモードはこれだよん")
            if self.opt.sb_mode == "both":
                self.compute_E_loss().backward()
                self._log_grad_stats('data_dependent_after_E_backward')
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F =  torch.optim.AdamW(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        if self.should_train_D:
            self.netD.train()
        if self.opt.sb_mode == 'both':
            self.netE.train()

        # OT-GAN 流の G/D 排他交互更新 (--n_gen)。
        # n_gen=0(既定)は従来どおり毎 step D→G の両方を更新。
        # n_gen=N>0 では (step+1)%N==0 の step は D のみ、それ以外は G のみ更新する
        # (OT-GAN train.py の意図 (i+1)%n_gen==0 と同じ。比率 G:D=(N-1):1)。
        n_gen = getattr(self.opt, 'n_gen', 0)
        _step = getattr(self, '_gd_step', 0)
        self._gd_step = _step + 1
        if n_gen and n_gen > 0:
            update_D_now = self.should_train_D and ((_step + 1) % n_gen == 0)
            update_G_now = not update_D_now
        else:
            update_D_now = self.should_train_D
            update_G_now = True

        # update D: gan=fake_B を偽物、real_B を本物と見抜くよう学習 /
        #           ot =D 特徴空間の Sinkhorn divergence を最大化
        if update_D_now:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            # Discriminator に勾配が流れているか確認
            self._log_grad_stats('optimize_after_D_backward')
            # netD パラメータに流れた勾配統計（adv ロスのみ由来）を wandb 用に記録
            d_grads = [p.grad.detach().flatten() for p in self.netD.parameters() if p.grad is not None]
            if d_grads:
                all_d_grads = torch.cat(d_grads)
                self.loss_NETD_GRAD_NORM = all_d_grads.norm().item()
                self.loss_NETD_GRAD_MIN  = all_d_grads.min().item()
                self.loss_NETD_GRAD_MAX  = all_d_grads.max().item()
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

        # n_gen>0 の D-only step はここで終了（G は次 step 以降で更新）
        if not update_G_now:
            return

        # update G
        self.set_requires_grad(self.netD, False)
        if self.opt.sb_mode == 'both':
            self.set_requires_grad(self.netE, False)
        
        self.optimizer_G.zero_grad()
        # if self.opt.netF == 'mlp_sample':
        #     self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        # GAN由来 / SB由来 の netG パラメータ勾配ノルムを分離ログ。
        # 追加 backward が要るので grad_stats_freq 間引き。retain_graph=True で本 backward は温存。
        # autograd.grad は .grad を汚さない（=optimizer step には影響しない）。AMP前の真の(unscaled)勾配。
        if self.opt.sb_mode in ['seq_ot', 'both']:
            _freq = getattr(self.opt, 'grad_stats_freq', 100)
            if _freq > 0 and getattr(self, '_grad_stats_step', 0) % _freq == 0:
                _params = [p for p in self.netG.parameters() if p.requires_grad]
                def _grad_norm_of(_loss):
                    if not torch.is_tensor(_loss) or not _loss.requires_grad:
                        return 0.0
                    gs = torch.autograd.grad(_loss, _params, retain_graph=True, allow_unused=True)
                    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in gs if g is not None)))
                self.loss_NETG_GRAD_GAN_NORM = _grad_norm_of(self.loss_G_GAN)
                _sb_term = (self.opt.lambda_SB * self.loss_SB) if torch.is_tensor(self.loss_SB) else 0.0
                self.loss_NETG_GRAD_SB_NORM = _grad_norm_of(_sb_term)
        if self.amp_netg:
            # netG が fp16 のため勾配を scale して backward、step 前に unscale
            self.scaler.scale(self.loss_G).backward()
            self.scaler.unscale_(self.optimizer_G)
        else:
            self.loss_G.backward()
        # Generator / Feature netF に勾配が流れているか確認
        self._log_grad_stats('optimize_after_G_backward')
        # netG パラメータの勾配統計（unscale 済みの実勾配を参照）
        if self.opt.sb_mode in ['seq_ot', 'both']:
            grads = [p.grad.detach().flatten() for p in self.netG.parameters() if p.grad is not None]
            if grads:
                all_grads = torch.cat(grads)
                self.loss_NETG_GRAD_NORM = all_grads.norm().item()
                self.loss_NETG_GRAD_MIN  = all_grads.min().item()
                self.loss_NETG_GRAD_MAX  = all_grads.max().item()
        if self.amp_netg:
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            self.optimizer_G.step()
        # if self.opt.netF == 'mlp_sample':
        #     self.optimizer_F.step()       
        
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

                Xt_1 = self._run_netG(Xt_in, time_idx, z)
                Xt_1 = self._restore_seq_from_net(Xt_1, Xt_meta)

                Xt2 = self.real_A2 if (t == 0) else (1 - inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)

                Xt2_in, Xt2_meta = self._flatten_seq_for_net(Xt2)
                n_xt2 = Xt2_in.shape[0]

                time_idx = torch.full((n_xt2,), t, device=self.real_A.device, dtype=torch.long)
                z = torch.randn(size=[n_xt2, 4 * self.opt.ngf], device=self.real_A.device)

                Xt_12 = self._run_netG(Xt2_in, time_idx, z)
                Xt_12 = self._restore_seq_from_net(Xt_12, Xt2_meta)
                
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1 - inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)

                    XtB_in, XtB_meta = self._flatten_seq_for_net(XtB)
                    n_xtb = XtB_in.shape[0]

                    time_idx = torch.full((n_xtb,), t, device=self.real_A.device, dtype=torch.long)
                    z = torch.randn(size=[n_xtb, 4 * self.opt.ngf], device=self.real_A.device)

                    Xt_1B = self._run_netG(XtB_in, time_idx, z)
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

        self.fake = self._run_netG(realt_in, time_idx_realt, z_in)
        self.fake = self._restore_seq_from_net(self.fake, realt_meta)

        realA2_in, realA2_meta = self._flatten_seq_for_net(self.real_A_noisy2)
        n_realA2 = realA2_in.shape[0]
        time_idx_realA2 = torch.full((n_realA2,), int(self.time_idx[0]), device=self.real_A.device, dtype=torch.long)
        z_in2 = torch.randn(size=[n_realA2, 4 * self.opt.ngf], device=self.real_A.device)

        self.fake_B2 = self._run_netG(realA2_in, time_idx_realA2, z_in2)
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

                    Xt_1 = self._run_netG(Xt_in, time_idx, z)
                    Xt_1 = self._restore_seq_from_net(Xt_1, Xt_meta)

                    setattr(self, "fake_"+str(t+1), Xt_1)
                    
    def compute_D_loss(self):
        """Calculate loss for the discriminator (gan: patchGAN / ot: OT-GAN流 divergence 最大化)"""
        if self.d_loss_mode == 'ot':
            return self.compute_D_loss_ot()
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

    def compute_D_loss_ot(self):
        """OT-GAN 流の D loss: D 特徴空間の seq-OT Sinkhorn divergence を最大化する。

        loss_D = -mean_seq SinkhornDiv(feat(fake_B.detach), feat(real_B))
        divergence = OT(f,r) - 0.5*OT(f,f) - 0.5*OT(r,r) なので、OT-GAN critic の
        「クロス項を最大化しつつ自己項(同一分布内の距離)を最小化する」構造と同等。
        G 側は既存の lambda_SB × seq-OT(flatten/random_patch) が同じ距離を最小化する
        側に回る (OT-GAN の loss.backward(mone) による勾配上昇に対応)。
        monotone / P_entropy / 手作り divergence は G のフレーム順序づけ用の項なので
        D 側では使わず、距離(divergence)のみを最大化する。
        """
        fake = self.fake_B.detach()
        real = self.real_B

        # netD に勾配が流れる状態で fake/real 両方の penultimate 特徴を取る
        fake_use, fake_meta = self._flatten_seq_for_net(fake)
        real_use, real_meta = self._flatten_seq_for_net(real)
        _, feat_fake = self.netD(fake_use, self.time_idx, return_feat=True)
        _, feat_real = self.netD(real_use, self.time_idx, return_feat=True)
        feat_fake_seq = self._restore_feat_seq(feat_fake, fake_meta)
        feat_real_seq = self._restore_feat_seq(feat_real, real_meta)

        # (B, T, ...) に揃える（4D 入力は 1 シーケンス扱い）
        if fake.dim() == 4:
            fake_seq = fake.unsqueeze(0)
            real_seq = real.unsqueeze(0)
            feat_fake_seq = feat_fake_seq.unsqueeze(0)
            feat_real_seq = feat_real_seq.unsqueeze(0)
        else:
            fake_seq = fake
            real_seq = real

        # random_patch は G 側と同様、fake/real 共通の空間位置を 1 つ選ぶ
        hw = None
        if getattr(self.opt, 'seq_ot_cost_space', 'pixel') == 'random_patch':
            fh, fw = feat_fake_seq.shape[-2], feat_fake_seq.shape[-1]
            hw = (int(torch.randint(fh, size=(1,))), int(torch.randint(fw, size=(1,))))

        b = fake_seq.shape[0]
        div_total = 0.0
        for i in range(b):
            ff = self._seq_descriptor(feat_fake_seq[i], hw)
            tf = self._seq_descriptor(feat_real_seq[i], hw)
            # debias=True 固定 (Sinkhorn divergence)。valid_idx 判定のため画像も渡す。
            ot_val, _ = sequence_ot_loss(
                fake_seq[i],
                real_seq[i],
                fake_feat=ff,
                tgt_feat=tf,
                solver='geo',
                reg=self.opt.lmda,
                iters=getattr(self.opt, 'seq_ot_iters', 50),
                monotone=False,
                P_entropy=False,
                ot_divergence=False,
                normalize=None,
                geo_scaling=getattr(self.opt, 'seq_ot_geo_scaling', 0.99),
                geo_p=getattr(self.opt, 'seq_ot_geo_p', 2),
                unbalanced=getattr(self.opt, 'seq_ot_unbalanced', None),
                debias=True,
                metric=getattr(self.opt, 'seq_ot_metric', 'l2'),
                feat_norm=getattr(self.opt, 'seq_ot_feat_norm', 'none'),
            )
            div_total = div_total + ot_val

        # D は divergence を最大化する（OT-GAN の gradient ascent 相当で符号反転）
        self.loss_D = -(div_total / b)
        self.loss_D_real = 0.0
        self.loss_D_fake = 0.0
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
        
        # D 特徴空間 OT 用の特徴記述子（pixel モードでは None）。
        # adv 有効時は GAN loss と同じ forward から特徴を取り出して二度 forward を避ける。
        # seq-OT が実際に走るときだけ特徴を取る（無駄な D forward を避ける）。
        need_feat = (self.use_D_feat_ot and self.opt.lambda_SB > 0.0
                     and self.opt.sb_mode in ('seq_ot', 'both'))
        self._feat_fake_seq = None
        self._feat_tgt_seq = None
        if self.opt.lambda_GAN > 0.0:
            # D には 4D テンソルを渡す
            fake_use, fake_meta = self._flatten_seq_for_net(fake)
            if need_feat:
                pred_fake, feat_fake = self.netD(fake_use, self.time_idx, return_feat=True)
                self._feat_fake_seq = self._restore_feat_seq(feat_fake, fake_meta)
            else:
                pred_fake = self.netD(fake_use, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
            # adv を使わないが D 特徴 OT を使う場合は、特徴取得のため1回だけ forward
            if need_feat:
                fake_use, fake_meta = self._flatten_seq_for_net(fake)
                _, feat_fake = self.netD(fake_use, self.time_idx, return_feat=True)
                self._feat_fake_seq = self._restore_feat_seq(feat_fake, fake_meta)
        # target 特徴は固定なので no_grad（detach）で取得
        if need_feat:
            with torch.no_grad():
                real_use, real_meta = self._flatten_seq_for_net(self.real_B)
                _, feat_tgt = self.netD(real_use, self.time_idx, return_feat=True)
                self._feat_tgt_seq = self._restore_feat_seq(feat_tgt, real_meta)
        self.loss_SB = 0
        self.loss_SB_P = 0.0
        self.loss_SB_U = 0.0
        self.loss_SB_ENT = 0.0
        self.loss_SB_DIV = 0.0
        self.loss_M_SUM = 0.0
        self.loss_M_DIAG = 0.0
        self.loss_P_IDENT_ERR = 0.0
        if getattr(self.opt, 'seq_ot_debias', False):
            self.loss_SB_GT = 0.0
            self.loss_SB_GG = 0.0
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
        
        # UNSB の SB ドリフト項 (tau·‖Xt - fake_B‖²) を独立に加える。
        # ログ用の self.loss_MSE は生の mean((Xt-fake)^2)(tau が小さく tau*mean だと
        # 0 に丸まって監視しづらいため)。UNSB と同じ tau 係数は loss_G 側で掛け、
        # lambda_MSE=1 で compute_original_sb_loss(both)内の同項と同じ寄与になる。
        self.loss_MSE = 0.0
        if getattr(self.opt, 'lambda_MSE', 0.0) > 0.0:
            self.loss_MSE = torch.mean((self.real_A_noisy - self.fake_B) ** 2)

        self.loss_G = (self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB
                       + self.opt.lambda_NCE*loss_NCE_both
                       + self.opt.lambda_MSE*self.opt.tau*self.loss_MSE)
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
        P_d = details.get('P_d')
        M_d = details.get('M_d')
        valid_idx = details.get('valid_idx')
        ot_cost = details.get('ot_cost')
        mono_cost = details.get('mono_cost')
        entropy_cost = details.get('entropy_cost')
        ot_divergence_cost = details.get('ot_divergence_cost')
        mono_loss = details.get('mono_loss')
        P_entropy_loss = details.get('P_entropy_loss')
        ot_divergence_loss = details.get('ot_divergence_loss')
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
            P_d=P_d.detach().cpu().numpy() if P_d is not None else None,
            M_d=M_d.detach().cpu().numpy() if M_d is not None else None,
            valid_idx=valid_idx.detach().cpu().numpy() if valid_idx is not None else None,
            ot_cost=float(ot_cost.detach().cpu().item()) if ot_cost is not None else None,
            mono_cost=float(mono_cost.detach().cpu().item()) if mono_cost is not None else None,
            entropy_cost=float(entropy_cost.detach().cpu().item()) if entropy_cost is not None else None,
            ot_divergence_cost=float(ot_divergence_cost.detach().cpu().item()) if ot_divergence_cost is not None else None,
            mono_loss=float(mono_loss.detach().cpu().item()) if mono_loss is not None else None,
            P_entropy_loss=float(P_entropy_loss.detach().cpu().item()) if P_entropy_loss is not None else None,
            ot_divergence_loss=float(ot_divergence_loss.detach().cpu().item()) if ot_divergence_loss is not None else None,
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
                if mono_cost is not None:
                    f.write(f'mono_cost (monotone weighted): {float(mono_cost.detach().cpu().item()):.6f}\n')
                if entropy_cost is not None:
                    f.write(f'entropy_cost (P entropy weighted): {float(entropy_cost.detach().cpu().item()):.6f}\n')
                if ot_divergence_cost is not None:
                    f.write(f'ot_divergence_cost (divergence weighted): {float(ot_divergence_cost.detach().cpu().item()):.6f}\n')
                if mono_loss is not None:
                    f.write(f'mono_loss (raw): {float(mono_loss.detach().cpu().item()):.6f}\n')
                if P_entropy_loss is not None:
                    f.write(f'P_entropy_loss (raw): {float(P_entropy_loss.detach().cpu().item()):.6f}\n')
                if ot_divergence_loss is not None:
                    f.write(f'ot_divergence_loss (raw): {float(ot_divergence_loss.detach().cpu().item()):.6f}\n')
                if total is not None:
                    f.write(f'total (distance + mono + entropy + divergence): {float(total.detach().cpu().item()):.6f}\n')
                if (ot_cost is not None) or (mono_cost is not None) or (entropy_cost is not None) or (ot_divergence_cost is not None) or (mono_loss is not None) or (P_entropy_loss is not None) or (ot_divergence_loss is not None) or (total is not None):
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

            # ot_divergence=True のときのみ P_d / M_d が details に入る
            if P_d is not None:
                P_d_cpu = P_d.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(P_d_cpu, aspect='auto', origin='lower')
                ax.set_xlabel('fake frame index')
                ax.set_ylabel('fake frame index')
                ax.set_title('transport plan P_d')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(base + '_P_d.png')
                plt.close(fig)

            if M_d is not None:
                M_d_cpu = M_d.detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(M_d_cpu, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('fake frame index')
                ax.set_ylabel('fake frame index')
                ax.set_title('cost matrix M_d')
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(base + '_M_d.png')
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

            # source(real_A) / target(real_B) / generated(fake_B) を
            # 縦に並べて 1 枚で見比べられる比較画像を保存する
            def _seq_to_canvas(seq_arr):
                """seq_arr: (T, C, H, W) or (T, H, W) を [0,1] の (H, W*T) or
                (H, W*T, C) キャンバスに変換して返す。失敗時は None。"""
                if seq_arr is None:
                    return None
                arr = _np.array(seq_arr)
                if arr.ndim == 4:      # (T, C, H, W)
                    T, C, H, W = arr.shape
                elif arr.ndim == 3:    # (T, H, W)
                    T, H, W = arr.shape
                    C = 1
                    arr = arr[:, _np.newaxis, ...]
                else:
                    return None
                arr = _np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
                if C == 1:
                    canvas = _np.zeros((H, W * T), dtype=_np.float32)
                    for t in range(T):
                        canvas[:, t * W:(t + 1) * W] = arr[t, 0]
                else:
                    canvas = _np.zeros((H, W * T, C), dtype=_np.float32)
                    for t in range(T):
                        canvas[:, t * W:(t + 1) * W, :] = arr[t].transpose(1, 2, 0)
                return canvas

            rows = [
                ('source (real_A)', _seq_to_canvas(real_A_arr)),
                ('target (real_B)', _seq_to_canvas(real_B_arr)),
                ('generated (fake_B)', _seq_to_canvas(fake_B_arr)),
            ]
            rows = [(title, canvas) for title, canvas in rows if canvas is not None]
            if rows:
                # フレーム数（横幅）からおおよその figsize を決める
                first = rows[0][1]
                n_frames = first.shape[1] / max(first.shape[0], 1)
                fig, axes = plt.subplots(
                    len(rows), 1,
                    figsize=(max(1.5 * n_frames, 4), 3 * len(rows)),
                )
                if len(rows) == 1:
                    axes = [axes]
                for ax, (title, canvas) in zip(axes, rows):
                    if canvas.ndim == 2:
                        ax.imshow(canvas, cmap='gray', vmin=0.0, vmax=1.0)
                    else:
                        ax.imshow(canvas, vmin=0.0, vmax=1.0)
                    ax.set_axis_off()
                    ax.set_title(title)
                fig.tight_layout()
                fig.savefig(base + '_seq_compare.png')
                plt.close(fig)
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

    def _restore_feat_seq(self, feat, meta):
        """netD penultimate 特徴 (B*T, C', H', W') を (B, T, C', H', W') に戻す。

        meta は _flatten_seq_for_net の戻り値（画像基準の (b,t,c,h,w) か None）。
        特徴は空間サイズ C',H',W' が画像と異なるので b,t だけ流用し、残りは feat の形を使う。
        meta=None（4D 入力）のときは feat 自体を 1 シーケンス (T,C',H',W') とみなしてそのまま返す。
        """
        if meta is None:
            return feat
        b, t = meta[0], meta[1]
        c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
        return feat.reshape(b, t, c, h, w)

    def _seq_descriptor(self, feat_seq, hw):
        """1 シーケンスの penultimate 特徴 (T, C', H', W') をフレーム記述子 (T, D) に縮約する。

        flatten      : 全空間位置を 1 ベクトルに -> (T, C'*H'*W')
        random_patch : 指定の空間位置 (h,w) の特徴のみ -> (T, C')  (hw は fake/tgt 共通=対応位置)
        """
        mode = getattr(self.opt, 'seq_ot_cost_space', 'pixel')
        if mode == 'flatten':
            return feat_seq.reshape(feat_seq.shape[0], -1)
        elif mode == 'random_patch':
            h, w = hw
            return feat_seq[:, :, h, w]
        else:
            raise ValueError(f"_seq_descriptor called with cost_space={mode}")

    def _extract_D_feat_seq(self, seq_bt):
        """1 シーケンス seq_bt: (T,C,H,W) を netD に通して penultimate 特徴
        (T,C',H',W') を返す。use_D_feat_ot=False（pixel モード）のときは None。

        学習時 compute_G_loss と同じ手順（_flatten_seq_for_net -> netD(return_feat)
        -> _restore_feat_seq）で特徴を取り出すためのヘルパー。可視化
        (save_ot_snapshots) で学習と同じ D 特徴コストを再現するのに使う。
        no_grad 前提の呼び出しを想定。
        """
        if not getattr(self, 'use_D_feat_ot', False):
            return None
        flat, meta = self._flatten_seq_for_net(seq_bt)   # (T,C,H,W) -> (T,C,H,W), meta=None
        _, feat = self.netD(flat, self.time_idx, return_feat=True)
        return self._restore_feat_seq(feat, meta)        # meta=None -> feat をそのまま (T,C',H',W')

    def seq_ot_descriptors(self, fake_seq_bt, real_seq_bt):
        """fake/target の 1 シーケンス (T,C,H,W) から seq-OT 用フレーム記述子
        (ff, tf) = (T,D),(T,D) を返す。pixel モードや use_D_feat_ot=False では
        (None, None) を返し、呼び出し側は従来どおり pixel コストにフォールバックする。

        random_patch のときは compute_sequence_ot_loss と同様、fake/tgt 共通の
        空間位置 hw を1つ選んで両者に使う（=対応位置）。
        """
        if not getattr(self, 'use_D_feat_ot', False):
            return None, None
        fake_feat_seq = self._extract_D_feat_seq(fake_seq_bt)
        tgt_feat_seq = self._extract_D_feat_seq(real_seq_bt)
        hw = None
        if getattr(self.opt, 'seq_ot_cost_space', 'pixel') == 'random_patch':
            fh, fw = fake_feat_seq.shape[-2], fake_feat_seq.shape[-1]
            hw = (int(torch.randint(fh, size=(1,))), int(torch.randint(fw, size=(1,))))
        ff = self._seq_descriptor(fake_feat_seq, hw)
        tf = self._seq_descriptor(tgt_feat_seq, hw)
        return ff, tf

    def _tsne_frame_descriptors(self, seq_bt):
        """1 シーケンス (T,C,H,W) を netD penultimate 特徴のフレーム記述子 (T, D) numpy にする。

        seq_ot_descriptors と違い cost_space に依らず常に flatten で取り出す
        （= D が距離を測っている penultimate 特徴空間そのものを可視化するため）。
        netD が無い場合は None。no_grad 前提。
        """
        if not hasattr(self, 'netD'):
            return None
        flat, meta = self._flatten_seq_for_net(seq_bt)          # (T,C,H,W), meta=None
        _, feat = self.netD(flat, self.time_idx, return_feat=True)
        feat = self._restore_feat_seq(feat, meta)               # (T,C',H',W')
        return feat.reshape(feat.shape[0], -1).detach().cpu().numpy()

    def save_D_feature_tsne(self, samples, save_path, title=None, draw_pair_links=True):
        """netD の特徴空間を t-SNE で 2D 可視化して PNG 保存する。

        samples: list of dict
            {'case': str, 'ct': (T,C,H,W) tensor(real_A), 'mr': (T,C,H,W) tensor(real_B)}
        可視化ルール:
            形 (marker)  = モダリティ  CT='^'(三角), MR='o'(丸)
            色 (color)   = ペア/患者(case)  … 同色の三角と丸が同一患者の CT/MR ペア
            細い線        = 同一 case・同一フレーム idx の CT-MR ペアを結ぶ（対応の可視化）
        学習では pair 情報を使っていないが、D が CT/MR を分離しつつ同一患者を
        近くに寄せているか（=ペアが分かるか）を目で確認するための図。
        """
        if not hasattr(self, 'netD'):
            print('[tsne] netD が無いためスキップします')
            return
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except Exception as e:
            print(f'[tsne] 依存ライブラリの読み込みに失敗: {e}')
            return

        feats, mod, pair, frame = [], [], [], []   # mod: 0=CT,1=MR / pair: case idx / frame: slice idx
        case_names = []
        with torch.no_grad():
            for s in samples:
                ct = s.get('ct'); mr = s.get('mr')
                if ct is None or mr is None:
                    continue
                ct_d = self._tsne_frame_descriptors(ct)
                mr_d = self._tsne_frame_descriptors(mr)
                if ct_d is None or mr_d is None:
                    continue
                pid = len(case_names)
                case_names.append(str(s.get('case', pid)))
                for d, m in ((ct_d, 0), (mr_d, 1)):
                    for t in range(d.shape[0]):
                        feats.append(d[t]); mod.append(m); pair.append(pid); frame.append(t)

        if len(feats) < 3:
            print(f'[tsne] 点が少なすぎるためスキップ (N={len(feats)})')
            return

        X = np.asarray(feats, dtype=np.float32)
        mod = np.asarray(mod); pair = np.asarray(pair); frame = np.asarray(frame)
        N = X.shape[0]

        # 高次元なので PCA で前処理してから t-SNE
        n_pca = int(min(50, N - 1, X.shape[1]))
        if n_pca >= 2:
            X = PCA(n_components=n_pca, random_state=0).fit_transform(X)
        perp = float(max(2.0, min(getattr(self.opt, 'tsne_perplexity', 30.0), (N - 1) / 3.0)))
        emb = TSNE(n_components=2, perplexity=perp, init='pca',
                   learning_rate='auto', random_state=0).fit_transform(X)

        n_cases = len(case_names)
        cmap = plt.get_cmap('tab20' if n_cases > 10 else 'tab10')
        colors = [cmap(i % cmap.N) for i in range(n_cases)]

        fig, ax = plt.subplots(figsize=(9, 8))

        # 同一 case・同一フレームの CT-MR ペアを細い線で結ぶ
        if draw_pair_links:
            for pid in range(n_cases):
                for t in np.unique(frame[pair == pid]):
                    ci = np.where((pair == pid) & (frame == t) & (mod == 0))[0]
                    mi = np.where((pair == pid) & (frame == t) & (mod == 1))[0]
                    if ci.size and mi.size:
                        ax.plot([emb[ci[0], 0], emb[mi[0], 0]],
                                [emb[ci[0], 1], emb[mi[0], 1]],
                                color=colors[pid], alpha=0.20, linewidth=0.6, zorder=1)

        for m, marker, name in ((0, '^', 'CT'), (1, 'o', 'MR')):
            sel = mod == m
            ax.scatter(emb[sel, 0], emb[sel, 1],
                       c=[colors[p] for p in pair[sel]],
                       marker=marker, s=45, edgecolors='k', linewidths=0.3,
                       alpha=0.9, zorder=2, label=name)

        # 凡例1: 形 = モダリティ（フォントに日本語が無いため英語表記）
        shape_leg = [Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                            markeredgecolor='k', markersize=9, label='CT (triangle)'),
                     Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                            markeredgecolor='k', markersize=9, label='MR (circle)')]
        leg1 = ax.legend(handles=shape_leg, title='shape = modality',
                         loc='upper left', fontsize=9)
        ax.add_artist(leg1)
        # 凡例2: 色 = ペア(患者/case)
        color_leg = [Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[i],
                            markeredgecolor='k', markersize=9, label=case_names[i])
                     for i in range(n_cases)]
        ax.legend(handles=color_leg, title='color = pair (patient)',
                  loc='upper right', fontsize=8, ncol=1)

        ax.set_title(title or 'netD feature space (t-SNE)')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        fig.tight_layout()
        try:
            fig.savefig(save_path, dpi=150)
            print(f'[tsne] saved: {save_path}  (N={N}, cases={n_cases}, perplexity={perp:.1f})')
        finally:
            plt.close(fig)

    def _run_netG(self, x, time_idx, z):
        """netG を frame 軸 (dim0 = flatten した B*T) でチャンク実行する。

        volume を 1 sequence として扱うと x は (B*T, C, H, W) で B*T が大きく
        (~200, nce_idt で ~400)、一括で conv/pad に通すと中間テンソルが
        32bit index 上限を超えて落ちる。chunk 枚ずつ netG に通して結合する。
        frame の間引きは一切せず全フレームを処理するので、後段 seq-OT の
        対応学習には影響しない。autograd も連結を通して通常どおり流れる。

        chunk<=0 もしくは枚数が chunk 以下なら従来どおり一括実行。
        """
        chunk = getattr(self.opt, 'netg_chunk', 0)
        n = x.shape[0]
        use_ckpt = (getattr(self.opt, 'netg_checkpoint', False)
                    and self.opt.isTrain and torch.is_grad_enabled())
        use_amp = getattr(self.opt, 'amp_netg', False)

        def _call(xc, tc, zc):
            with autocast(enabled=use_amp):
                if use_ckpt:
                    out = checkpoint(self.netG, xc, tc, zc, use_reentrant=False)
                else:
                    out = self.netG(xc, tc, zc)
            # 下流(seq-OT / D / E)を fp32 に保つため出力を戻す
            return out.float() if use_amp else out

        if not chunk or chunk <= 0 or n <= chunk:
            return _call(x, time_idx, z)
        outs = []
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            outs.append(_call(x[sl], time_idx[sl], z[sl]))
        return torch.cat(outs, dim=0)

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

        # D 特徴空間 OT 用の特徴記述子。pixel モードでは None（画像から従来どおりコスト計算）。
        # feat は compute_G_loss で抽出済み（adv と forward 共有）。
        fake_feat_seq = getattr(self, '_feat_fake_seq', None)
        tgt_feat_seq = getattr(self, '_feat_tgt_seq', None)
        use_feat = self.use_D_feat_ot and (fake_feat_seq is not None) and (tgt_feat_seq is not None)
        # random_patch は毎 step 1 空間位置を選ぶ（バッチ・fake/tgt 共通=対応位置）
        hw = None
        if use_feat and getattr(self.opt, 'seq_ot_cost_space', 'pixel') == 'random_patch':
            fh, fw = fake_feat_seq.shape[-2], fake_feat_seq.shape[-1]
            hw = (int(torch.randint(fh, size=(1,))), int(torch.randint(fw, size=(1,))))

        # sequence OT がどのテンソルから構成されていて、
        if self._grad_debug_enabled:
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
                pass

        if fake_seq.dim() == 5 and real_seq.dim() == 5:
            b, t, c, h, w = fake_seq.shape
            seq_ot = 0.0
            seq_ot_dist = 0.0
            seq_ot_reg = 0.0
            seq_ot_entropy = 0.0
            seq_ot_divergence = 0.0
            seq_ot_gen_tgt = 0.0
            seq_ot_gen_gen = 0.0
            seq_ot_msum = 0.0
            seq_ot_mdiag = 0.0
            seq_ot_pident = 0.0

            for i in range(b):
                ff = self._seq_descriptor(fake_feat_seq[i], hw) if use_feat else None
                tf = self._seq_descriptor(tgt_feat_seq[i], hw) if use_feat else None
                ot_val, terms = sequence_ot_loss(
                    fake_seq[i],
                    real_seq[i],
                    fake_feat=ff,
                    tgt_feat=tf,
                    solver=getattr(self.opt, 'seq_ot_solver', 'pot'),
                    reg=self.opt.lmda,
                    iters=getattr(self.opt, 'seq_ot_iters', 50),
                    monotone=getattr(self.opt, 'seq_ot_monotone', True),
                    monotone_penalty=getattr(self.opt, 'seq_ot_monotone_penalty', 1.0),
                    P_entropy=getattr(self.opt, 'seq_ot_p_entropy', True),
                    P_entropy_penalty=getattr(self.opt, 'seq_ot_p_entropy_penalty', 1.0),
                    ot_divergence=getattr(self.opt, 'seq_ot_divergence', False),
                    ot_divergence_penalty=getattr(self.opt, 'seq_ot_divergence_penalty', -0.5),
                    normalize=(None if getattr(self.opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(self.opt, 'seq_ot_normalize', 'mean')),
                    return_details=False,
                    sinkhorn_type=getattr(self.opt, 'sinkhorn_type', 'sinkhorn'),
                    geo_scaling=getattr(self.opt, 'seq_ot_geo_scaling', 0.99),
                    geo_p=getattr(self.opt, 'seq_ot_geo_p', 2),
                    unbalanced=getattr(self.opt, 'seq_ot_unbalanced', None),
                    debias=getattr(self.opt, 'seq_ot_debias', False),
                    metric=getattr(self.opt, 'seq_ot_metric', 'l2'),
                    feat_norm=getattr(self.opt, 'seq_ot_feat_norm', 'none'),
                )

                seq_ot = seq_ot + ot_val
                if terms is not None:
                    seq_ot_dist = seq_ot_dist + terms['ot_cost']
                    seq_ot_reg = seq_ot_reg + terms['mono_cost']
                    seq_ot_entropy = seq_ot_entropy + terms['entropy_cost']
                    seq_ot_divergence = seq_ot_divergence + terms['ot_divergence_cost']
                    if terms.get('gen_tgt_cost') is not None:
                        seq_ot_gen_tgt = seq_ot_gen_tgt + terms['gen_tgt_cost']
                    if terms.get('gen_gen_cost') is not None:
                        seq_ot_gen_gen = seq_ot_gen_gen + terms['gen_gen_cost']
                    if terms.get('M_sum') is not None:
                        seq_ot_msum = seq_ot_msum + terms['M_sum']
                        seq_ot_mdiag = seq_ot_mdiag + terms['M_diag']
                        seq_ot_pident = seq_ot_pident + terms['P_ident_err']

            loss_seq = seq_ot / b
            self.loss_SB_P = seq_ot_dist / b
            self.loss_SB_U = seq_ot_reg / b
            self.loss_SB_ENT = seq_ot_entropy / b
            self.loss_SB_DIV = seq_ot_divergence / b
            self.loss_M_SUM = seq_ot_msum / b
            self.loss_M_DIAG = seq_ot_mdiag / b
            self.loss_P_IDENT_ERR = seq_ot_pident / b
            # debias 時のみ: gen→tgt / gen→gen を別々にログ（loss には未使用）
            if getattr(self.opt, 'seq_ot_debias', False):
                self.loss_SB_GT = seq_ot_gen_tgt / b
                self.loss_SB_GG = seq_ot_gen_gen / b
        else:
            ff = self._seq_descriptor(fake_feat_seq, hw) if use_feat else None
            tf = self._seq_descriptor(tgt_feat_seq, hw) if use_feat else None
            ot_val, terms = sequence_ot_loss(
                fake_seq,
                real_seq,
                fake_feat=ff,
                tgt_feat=tf,
                solver=getattr(self.opt, 'seq_ot_solver', 'pot'),
                reg=self.opt.lmda,
                iters=getattr(self.opt, 'seq_ot_iters', 50),
                monotone=getattr(self.opt, 'seq_ot_monotone', True),
                monotone_penalty=getattr(self.opt, 'seq_ot_monotone_penalty', 1.0),
                P_entropy=getattr(self.opt, 'seq_ot_p_entropy', True),
                P_entropy_penalty=getattr(self.opt, 'seq_ot_p_entropy_penalty', 1.0),
                ot_divergence=getattr(self.opt, 'seq_ot_divergence', False),
                ot_divergence_penalty=getattr(self.opt, 'seq_ot_divergence_penalty', -0.5),
                normalize=(None if getattr(self.opt, 'seq_ot_normalize', 'mean') == 'none' else getattr(self.opt, 'seq_ot_normalize', 'mean')),
                return_details=False,
                sinkhorn_type=getattr(self.opt, 'sinkhorn_type', 'sinkhorn'),
                geo_scaling=getattr(self.opt, 'seq_ot_geo_scaling', 0.99),
                geo_p=getattr(self.opt, 'seq_ot_geo_p', 2),
                unbalanced=getattr(self.opt, 'seq_ot_unbalanced', None),
                debias=getattr(self.opt, 'seq_ot_debias', False),
                metric=getattr(self.opt, 'seq_ot_metric', 'l2'),
                feat_norm=getattr(self.opt, 'seq_ot_feat_norm', 'none'),
            )

            loss_seq = ot_val
            if terms is not None:
                self.loss_SB_P = terms['ot_cost']
                self.loss_SB_U = terms['mono_cost']
                self.loss_SB_ENT = terms['entropy_cost']
                self.loss_SB_DIV = terms['ot_divergence_cost']
                if terms.get('M_sum') is not None:
                    self.loss_M_SUM = terms['M_sum']
                    self.loss_M_DIAG = terms['M_diag']
                    self.loss_P_IDENT_ERR = terms['P_ident_err']
                # debias 時のみ: gen→tgt / gen→gen を別々にログ（loss には未使用）
                if terms.get('gen_tgt_cost') is not None:
                    self.loss_SB_GT = terms['gen_tgt_cost']
                if terms.get('gen_gen_cost') is not None:
                    self.loss_SB_GG = terms['gen_gen_cost']

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

        # --- gradient stats (every grad_stats_freq steps to avoid per-step backward overhead) ---
        if self.isTrain:
            self._grad_stats_step += 1
            freq = getattr(self.opt, 'grad_stats_freq', 100)
            if freq > 0 and self._grad_stats_step % freq == 0:
                try:
                    g = torch.autograd.grad(loss_seq, fake_seq, retain_graph=True, allow_unused=True)[0]
                    if g is not None:
                        self.loss_SB_GRAD_NORM = g.norm().item()
                        self.loss_SB_GRAD_MIN  = g.min().item()
                        self.loss_SB_GRAD_MAX  = g.max().item()
                    else:
                        self.loss_SB_GRAD_NORM = 0.0
                        self.loss_SB_GRAD_MIN  = 0.0
                        self.loss_SB_GRAD_MAX  = 0.0
                    # monotone(U) 項だけの勾配を切り出してログ（penalty 込みの実寄与）。
                    # self.loss_SB_U は mono_cost(=penalty*mono_loss) のテンソル。グラフに繋がっていれば微分可。
                    mono_term = self.loss_SB_U
                    gu = None
                    if torch.is_tensor(mono_term) and mono_term.requires_grad:
                        gu = torch.autograd.grad(mono_term, fake_seq, retain_graph=True, allow_unused=True)[0]
                    if gu is not None:
                        self.loss_SB_U_GRAD_NORM = gu.norm().item()
                        self.loss_SB_U_GRAD_MIN  = gu.min().item()
                        self.loss_SB_U_GRAD_MAX  = gu.max().item()
                    else:
                        self.loss_SB_U_GRAD_NORM = 0.0
                        self.loss_SB_U_GRAD_MIN  = 0.0
                        self.loss_SB_U_GRAD_MAX  = 0.0
                except Exception:
                    pass

        grad_log_file = getattr(self.opt, 'grad_log_file', '')
        grad_log_epoch = getattr(self.opt, 'grad_log_epoch', -1)
        _should_log = grad_log_file and self.isTrain and (grad_log_epoch == -1 or self.current_epoch == grad_log_epoch)
        if _should_log:
            self._grad_log_iter += 1
            line = (
                f"reg={self.opt.lmda:.5f}"
                f" iter={self._grad_log_iter}"
                f" g_min={self.loss_SB_GRAD_MIN:.4e}"
                f" g_max={self.loss_SB_GRAD_MAX:.4e}"
                f" g_norm={self.loss_SB_GRAD_NORM:.4e}"
                f" loss={loss_seq.item():.6f}\n"
            )
            try:
                with open(grad_log_file, 'a') as _f:
                    _f.write(line)
            except Exception:
                pass

        return loss_seq
