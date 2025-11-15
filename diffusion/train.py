# !pip install matplotlib tqdm transformers kagglehub

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import itertools
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms 
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass

from transformers import get_cosine_schedule_with_warmup
from torchvision import datasets, transforms

import kagglehub

# Download latest version
path = kagglehub.dataset_download("splcher/animefacedataset")

IMAGE_FOLDER = path

class Sampler:
    """
    beta -> noise we are going to add at time step t.
    alpha -> 1 - beta 
    alpha_hat -> alpha_cumprod
    sigma_t -> std. dev from variance (which is beta_t_tilda) 
    """
    def __init__(self, num_steps = 1000, beta_start = 0.0001 , beta_end = 0.01):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta_schedule = self.linear_beta_scheduler()   # shape (num_steps,)

        self.alpha = 1 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim = -1)    # shape (num_steps,)

    def linear_beta_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps)
    
    # addding fake dim to match input to target
    def _repeat_unsqueeze(self, input, target):
        while input.dim() < target.dim():
            input = input.unsqueeze(-1)
        return input

    def add_noise(self, data, ts):
        device = data.device

        noise = torch.randn_like(data)
        alpha_hat = self.alpha_cumprod[ts].to(device)

        mean_coeff = alpha_hat ** 0.5
        var_coeff = (1 - alpha_hat) ** 0.5
        mean_coeff = self._repeat_unsqueeze(mean_coeff, data)
        var_coeff = self._repeat_unsqueeze(var_coeff, data)
        
        # add noise to data 
        noisy_data = mean_coeff * data + var_coeff * noise

        return noisy_data, noise

    def remove_noise(self, data, ts, predicted_noise):
        device = data.device

        noise = torch.randn_like(data)

        beta_t = self.beta_schedule[ts].to(device)
        alpha_t = self.alpha[ts].to(device)
        alpha_hat_t = self.alpha_cumprod[ts].to(device)
        alpha_hat_t_prev = self.alpha_cumprod[ts-1].to(device)
        
        ## for ts 0 there is no prev alpha -> here we set to 1.0
        ts_zero_mask = ( ts ==0 )
        alpha_hat_t_prev[ts_zero_mask] = 1.0

        ## sigma_t
        variance = ( (1 - alpha_hat_t_prev) * beta_t)/ (1 - alpha_hat_t)
        variance = self._repeat_unsqueeze(variance, data) 
        sigma_t_z = variance ** 0.5 * noise

        ## noise coeff
        noise_coeff = beta_t / ( 1 - alpha_hat_t ) ** 0.5
        noise_coeff = self._repeat_unsqueeze(noise_coeff, data)

        reciproc_root_alpha_t = alpha_t ** -0.5
        reciproc_root_alpha_t = self._repeat_unsqueeze(reciproc_root_alpha_t, data)

        # denoise image by mean
        mean = reciproc_root_alpha_t * ( data - noise_coeff * predicted_noise )
        denoised_data = mean + sigma_t_z

        return denoised_data

class SelfAttention(nn.Module):
    def __init__(self, channels=32, n_head=12, attn_p=0, proj_p=0):
        super().__init__()
        self.qkv = nn.Linear(channels, channels * 3)
        self.head_dim = channels // n_head
        self.n_head = n_head
        self.attn_p = attn_p
        self.proj = nn.Linear(channels, channels)
        self.proj_p = nn.Dropout(proj_p)

    def forward(self, x ):
        B, T, C = x.shape

        # '3' is there because we have combined qkv in same matrix
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # '3'(q, k, v) x B x H x T x E
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p)
        x = x.transpose(1,2).reshape(B, T, C)
        x = self.proj_p ( self.proj(x) )
        return x
    
class MLP(nn.Module):
    def __init__(self, channels=32, mlp_ratio=4, ff_drop=0):
        super().__init__()
        hddn_dim = channels * mlp_ratio
        self.proj_in = nn.Linear(channels, hddn_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hddn_dim, channels)
        self.ff_drop = nn.Dropout(ff_drop)

    def forward(self, x):
        x = self.proj_in(x)
        return self.ff_drop( self.proj_out( self.act(x) ) )

class TransformerBlock(nn.Module):
    def __init__(self, channels=32, n_head=8, mlp_ratio=4, attn_p=0, proj_p=0, ff_drop=0):
        super().__init__()
        self.sa = SelfAttention(channels=channels, n_head=n_head, attn_p=attn_p, proj_p=proj_p)
        self.mlp = MLP(channels=channels, mlp_ratio=mlp_ratio, ff_drop=ff_drop)
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp( self.ln2(x) )

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class SinusoidalTimesEmb(nn.Module):
    def __init__(self, t_emb_dim, scaled_t_emb_dim):
        super().__init__()
        self.inv_freq = nn.Parameter( 10000 ** (torch.arange(0, t_emb_dim, 2).float()/ t_emb_dim ), requires_grad=False )
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, scaled_t_emb_dim),
            nn.SiLU(),
            nn.Linear(scaled_t_emb_dim, scaled_t_emb_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # To get freqs for all T=t, add fake dims: ts[t, 1] * freq[1, T//2]
        ts_freq = x.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        embs = torch.cat( [torch.sin(ts_freq), torch.cos(ts_freq)], dim=-1 )
        embs = self.time_mlp(embs)
        return embs


class ResidualBlock(nn.Module):
    """
    H and W are same for input and output, channels sometimes changes
    """
    def __init__(self, in_channels, out_channels, norm_groups, t_emb_dim):
        super().__init__()

        ## convert any time's Tensor dim to out_channel size, so we can add with the image
        self.time_expand = nn.Linear(t_emb_dim, out_channels)

        ## convs
        self.norm1 = nn.GroupNorm(norm_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')

        self.norm2 = nn.GroupNorm(norm_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')

        ## resize input to have same channel as output so that we can add both, for res connection
        self.resize_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, t_embs):
        res_conn = x
        t_embs = self.time_expand(t_embs)

        x = self.conv1( F.silu(self.norm1(x) ) )
        x = x + t_embs[:, :, None, None] ## add X + time info
        x = self.conv2( F.silu(self.norm2(x) ) )

        x = x + self.resize_conv(res_conn) ## resize input to match ouput channels
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.upsample(x)
        # because we will always downsample by 2 and upsample by 2
        assert x.shape == (B, C, H * 2, W * 2)
        return x

class UNET(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 start_dim = 64,
                 dim_mults = (1, 2, 4),
                 res_block_per_group = 1, 
                 num_group_norms = 16,
                 t_emb_dim = 128,
                 ):
        super().__init__()

        self.input_channels = in_channels
        channels_list = [i * start_dim for i in dim_mults]
        
        """
        image -> n_res_block x (same dim handling) -> downsample (H, W) -> attn -> channel inc
        1. (C x H x W) -> (C x H//2 x W//2) -> attn -> (2*C x H//2 x W//2)
        2. (2*C x H//2 x W//2) -> (2*C x H//4 x W//4) -> attn -> (4*C x H//4 x W//4)
        ...
        ...
        """
        self.encoder_config = []
        for idx, d in enumerate(channels_list):
            for n in range(res_block_per_group):
                self.encoder_config.append( ( ((d, d), "residual") ) )
            
            self.encoder_config.append( ((d, d), "downsample") ) # img size will become half with stride = 2
            self.encoder_config.append( ((d,), "attention") )

            if idx < len(channels_list) - 1:
                self.encoder_config.append( ((d, channels_list[idx+1]), "residual") )

        self.bottleneck_config = []
        for _ in range(res_block_per_group):
            self.bottleneck_config.append( ((d,d), "residual") )
        
        ## reverse the encoder
        out_dim = channels_list[-1]
        revered_encoder_config = list(reversed(self.encoder_config))
        self.decoder_config = []
        
        """
        we want to mirror enc add output from enc to dec via res conn, hence input channels will increase.
        go through in reverse order and:
        1. if residual
          - concat res conn on top of the output channels
        2. if downsample
          - concat res conn on top of the output channels
          - upsample
        3. attention
          - just do attention
        """
        for idx, (dim, blk_type) in enumerate(revered_encoder_config):
            if blk_type != "attention":
                enc_in_channels, enc_out_channels = dim[0], dim[-1]
                # on the output add mirrored encoder's output channels and conv to mirrored input size
                self.decoder_config.append( ( (out_dim + enc_out_channels, enc_in_channels ), "residual" ) )

                if blk_type == "downsample":
                    self.decoder_config.append( ( (enc_in_channels, enc_in_channels), "upsample" ) )
                
                out_dim = enc_in_channels
            else:
                enc_in_channels = dim
                self.decoder_config.append( ((dim[0], ), "attention") )
        self.decoder_config.append(((channels_list[0]*2, channels_list[0]), "residual"))
        
        """
        print("## encoder::")
        for _ in self.encoder_config:
            print(_)

        print("## self.bottleneck_config::")
        for _ in self.bottleneck_config:
            print(_)

        print("## decoder::")
        for _ in self.decoder_config:
            print(_)
        """
        
        ## defining model 
        # convert img channels to 1st block's channel for input
        self.proj_in = nn.Conv2d(in_channels=self.input_channels, out_channels=channels_list[0], kernel_size=3, padding="same")

        # u-net blocks
        build_args = {"num_group_norms": num_group_norms, "t_emb_dim": t_emb_dim}
        self.encoder = self._build_block(self.encoder_config, args=build_args)
        self.bottleneck = self._build_block(self.bottleneck_config, args=build_args)
        self.decoder = self._build_block(self.decoder_config, args=build_args)

        # out proj 
        self.proj_out = nn.Conv2d(channels_list[0], self.input_channels, kernel_size=3, padding="same")

        
    def _build_block(self, config, args={}):
        block = nn.ModuleList()
        for (dim, blk_type) in config:
            in_ch, out_ch = dim[0], dim[-1] 
            if blk_type == "residual":
                block.append( ResidualBlock(in_channels=in_ch, out_channels=out_ch,
                                            norm_groups=args["num_group_norms"], t_emb_dim=args["t_emb_dim"]) )
            elif blk_type == "attention":
                block.append(
                    TransformerBlock(in_ch)
                )
            elif blk_type == "downsample":
                block.append( 
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                )
            elif blk_type == "upsample":
                block.append(
                    Upsample(in_ch, out_ch)
                )
            else:
                raise Exception(f"Unknown blk type {blk_type}")
        return block

    def forward(self, x, time_embeddings):
        residuals = []
        x = self.proj_in(x)
        residuals.append(x)

        for blk in self.encoder:
            if isinstance(blk, ResidualBlock):
                x = blk(x, time_embeddings)
                residuals.append(x)
            elif isinstance(blk, TransformerBlock):
                x = blk(x)
                residuals.append(x)
            elif isinstance(blk, nn.Conv2d):
                x = blk(x)
        
        for blk in self.bottleneck:
            x = blk(x, time_embeddings)

        for blk in self.decoder:
            if isinstance(blk, ResidualBlock):
                residual_tensor = residuals.pop()
                x  = torch.cat([x, residual_tensor], axis=1)
                x = blk(x, time_embeddings)
            else:
                x = blk(x)
        
        x = self.proj_out(x)
        return x

class Diffusion(nn.Module):
    def __init__(self, 
                 in_channels = 3,
                 start_dim = 64,
                 dim_mults = (1, 2, 4),
                 res_block_per_group = 1, 
                 num_group_norms = 16,
                 t_emb_dim = 128,
                 t_emb_ratio = 2,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.start_dim = start_dim
        self.dim_mults = dim_mults
        self.res_block_per_group = res_block_per_group
        self.num_group_norms = num_group_norms
        self.t_emb_dim = t_emb_dim
        self.t_emb_ratio = t_emb_ratio

        scaled_t_emb_dim = self.t_emb_dim * self.t_emb_ratio

        self.time_embs = SinusoidalTimesEmb(self.t_emb_dim, scaled_t_emb_dim = scaled_t_emb_dim )
        self.unet = UNET(
                 in_channels = self.in_channels,
                 start_dim = self.start_dim,
                 dim_mults = self.dim_mults,
                 res_block_per_group = self.res_block_per_group,
                 num_group_norms = self.num_group_norms,
                 t_emb_dim = scaled_t_emb_dim
        )

    def forward(self, noisy_img, time_steps):

        # get time embs for given ts and pass to unet
        t_embs = self.time_embs(time_steps)

        pred_noise = self.unet(noisy_img, t_embs)

        return pred_noise

def save_model(m, path_to_generated_dir, step_idx):
    # Save model
    torch.save({
        'model_state_dict': m.state_dict()
    }, f"{path_to_generated_dir}/checkpoint_step_{step_idx}.pt")
    print(f"{path_to_generated_dir}/checkpoint_step_{step_idx}.pt")
    
@torch.no_grad()
def sample_plot_image(step_idx, 
                      total_timesteps, 
                      sampler, 
                      image_size,
                      num_channels,
                      plot_freq, 
                      model,
                      num_gens,
                      path_to_generated_dir,
                      device):

    ### Conver Tensor back to Image (From Huggingface Annotated Diffusion) ###
    tensor2image_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    images = torch.randn((num_gens, num_channels, image_size, image_size))
    num_images_per_gen = (total_timesteps // plot_freq)

    images_to_vis = [[] for _ in range(num_gens)]
    for t in np.arange(total_timesteps)[::-1]:
        ts = torch.full((num_gens, ), t)
        noise_pred = model(images.to(device), ts.to(device)).detach().cpu()
        images = sampler.remove_noise(images, ts, noise_pred)
        if t % plot_freq == 0:
            for idx, image in enumerate(images):
                images_to_vis[idx].append(tensor2image_transform(image))


    images_to_vis = list(itertools.chain(*images_to_vis))

    fig, axes = plt.subplots(nrows=num_gens, ncols=num_images_per_gen, figsize=(num_images_per_gen, num_gens))
    plt.tight_layout()
    for ax, image in zip(axes.ravel(), images_to_vis):
        ax.imshow(image)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(path_to_generated_dir, f"step_{step_idx}.png"))
    #plt.show()
    plt.close()
    
    save_model(model, path_to_generated_dir, step_idx)
    

def scale_to_minus1_1(t):
    return (t * 2) - 1

def train(image_size=64, 
          eval_interval=1000,
          total_timesteps=500, 
          plot_freq_interval=50, 
          num_generations=5, 
          num_training_steps=50000, 
          num_input_channels=3, 
          batch_size=64,
          path_to_generated="generated"):

    torch.backends.cudnn.benchmark = True
    os.makedirs(path_to_generated, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm_sampler = Sampler(num_steps=total_timesteps)
    loss_fn = nn.MSELoss()
    current_step = 0
    progress_bar = tqdm(range(num_training_steps))

    # Convert img to [-1, 1] tensor 
    img2tensor = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Lambda(scale_to_minus1_1)
                ])
    dataset = ImageFolder(IMAGE_FOLDER, transform=img2tensor) # https://www.kaggle.com/datasets/splcher/animefacedataset
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = Diffusion(in_channels=num_input_channels).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Parameters:", params)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0005)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=2500, 
                                                num_training_steps=num_training_steps)

    train = True
    while train:
        training_losses = []
        for images, _ in trainloader:
            batch_size = images.shape[0]
        
            timesteps = torch.randint(0,total_timesteps,(batch_size,)) # random sample T
            noisy_images, noise = ddpm_sampler.add_noise(images, timesteps) # add noise for T=t
            noise_pred = model(noisy_images.to(device), timesteps.to(device)) # model's noise pred
            loss = loss_fn(noise_pred, noise.to(device))

            training_losses.append(loss.cpu().item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            current_step += 1

            if (current_step % eval_interval == 0):
                loss_mean = np.mean(training_losses)
                print("Training loss:", loss_mean)
                print("Current learning rate:", optimizer.param_groups[-1]["lr"])

                training_losses = []
                print("Saving Image Generation")
                sample_plot_image(step_idx=current_step, 
                                  total_timesteps=total_timesteps, 
                                  sampler=ddpm_sampler, 
                                  image_size=image_size,
                                  num_channels=num_input_channels,
                                  plot_freq=plot_freq_interval, 
                                  model=model,
                                  num_gens=num_generations,
                                  path_to_generated_dir=path_to_generated,
                                  device=device)
                
            if current_step >= num_training_steps:
                print("Training Completed")
                train = False
                break
train()