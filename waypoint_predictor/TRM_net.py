import torch
import torch.nn as nn
import numpy as np
import utils
import argparse

from .transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig
from .img_depth_corss_attention import ID_CrossAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def TRM_predict(mode, args, predictor, rgb_feats, depth_feats,map_feats=None,cross_attn=False,occ=None):
    ''' predicting the waypoint probabilities '''
    vis_logits = predictor(rgb_feats, depth_feats,map_feats,pre_fuse=False,cross_attn=cross_attn)
    # entry-wise probabilities
    vis_probs = torch.sigmoid(vis_logits)

    if mode == 'train':
        return vis_logits
    elif mode == 'eval':
        return vis_probs, vis_logits


class BinaryDistPredictor_TRM(nn.Module):
    def __init__(self, args=None, hidden_dim=768, n_classes=12,batchsize=1):
        super(BinaryDistPredictor_TRM, self).__init__()
        self.args = args
        self.num_angles = 120
        self.num_imgs = 12
        self.n_classes = 12
        self.TRM_LAYER = 2
        self.TRM_NEIGHBOR = 1
        self.HEATMAP_OFFSET = 5

            
        self.n_classes = n_classes

        self.visual_fc_rgb = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(True),
        )

        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),
            nn.ReLU(True),
        )
        self.visual_merge = nn.Sequential(
            nn.Linear(int(hidden_dim*2), hidden_dim),
            nn.ReLU(True),
        )
        
        
        self.attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.ffn = nn.Sequential(
                nn.Linear(768, 3072),
                nn.ReLU(),
                nn.Dropout(0.1), 
                nn.Linear(3072, 768)
            )
        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        try:
            config.num_hidden_layers = args.TRM_LAYER
        except: 
            config.num_hidden_layers = self.TRM_LAYER
        # pre_fuse for cross attention between image and depth
        self.pre_fuse = ID_CrossAttention(embed_dim_img=int(hidden_dim/2), embed_dim_depth=hidden_dim, num_heads=config.num_attention_heads)
        self.waypoint_TRM = WaypointBert(config=config)


        layer_norm_eps = config.layer_norm_eps

        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=self.TRM_NEIGHBOR).to(device)
        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )

    def forward(self, rgb_feats, depth_feats,map_feats, pre_fuse=False,cross_attn=False):
        self.to("cuda")
        bsi = rgb_feats.size(0) // self.num_imgs
        rgb_x = rgb_feats.reshape(
            bsi, self.num_imgs, -1)

        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)
        # rgb_x = self.visual_1by1conv_rgb(rgb_feats) # rgb_feats.shape torch.Size([96, 768])
        if pre_fuse:
            # vis_x = self.pre_fuse(rgb_x,depth_x,self.num_imgs)
            vis_x =torch.cat((rgb_x, depth_x), dim=-1)
        elif cross_attn:
            vis_x = self.pre_fuse(rgb_x,depth_x,self.num_imgs)
        else:
            vis_x = self.visual_merge(
                torch.cat((rgb_x, depth_x), dim=-1)
            )


        attention_mask = self.mask.repeat(bsi,1,1,1)
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        )
        # # !
        vis_logits = self.vis_classifier(vis_rel_x)
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

        
        # heatmap offset (each image is pointing at the middle)
        try:
            vis_logits = torch.cat(
                (vis_logits[:,self.args.HEATMAP_OFFSET:,:], vis_logits[:,:self.args.HEATMAP_OFFSET,:]),
                dim=1)
        except:
            vis_logits = torch.cat(
            (vis_logits[:,self.HEATMAP_OFFSET:,:], vis_logits[:,:self.HEATMAP_OFFSET,:]),
            dim=1)
        return vis_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
