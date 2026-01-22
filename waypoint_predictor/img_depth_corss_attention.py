import torch
import torch.nn as nn
import torch.nn.functional as F

class ID_CrossAttention(nn.Module):
    def __init__(self, embed_dim_img, embed_dim_depth, num_heads, drop_prob=0.1):
        super(ID_CrossAttention, self).__init__()
        self.query_linear = nn.Linear(embed_dim_img, embed_dim_depth)
        self.key_linear = nn.Linear(embed_dim_depth, embed_dim_depth)
        self.value_linear = nn.Linear(embed_dim_depth, embed_dim_depth)
        self.multihead_attn = nn.MultiheadAttention(embed_dim_depth, num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = nn.LayerNorm(embed_dim_depth)
        
        # Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim_depth, embed_dim_depth * 4),
            nn.ReLU(),
            nn.Linear(embed_dim_depth * 4, embed_dim_depth)
        )
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(embed_dim_depth)
        
    def forward(self, img_feat, depth_feat, token_size):
        attn_mask = torch.zeros(token_size, token_size).to(img_feat.device)
        for i in range(token_size):
            attn_mask[i, (i - 1) % token_size] = 1  # Each image attends to the depth feature of the previous image
            attn_mask[i, i] = 1             # Each image attends to its own depth feature
            attn_mask[i, (i + 1) % token_size] = 1  # Each image attends to the depth feature of the next image
        attn_mask = ~attn_mask.bool()
        
        # Step 1: Project img_feat as the query, depth_feat as key and value
        batch_size, num_image, embed_dim_depth = depth_feat.size()
        query = self.query_linear(img_feat)  
        key = self.key_linear(depth_feat)    
        value = self.value_linear(depth_feat) 

        # Step 2: Apply multi-head attention with attention mask
        query = query.permute(1, 0, 2)  # [12, 1, 768]
        key = key.permute(1, 0, 2)      # [12, 1, 768]
        value = value.permute(1, 0, 2)  # [12, 1, 768]
        # ##play
        # aaa = nn.MultiheadAttention(embed_dim_depth, 8).to(img_feat.device)
        # aaa(query, key, value,attn_mask=attn_mask)
        # ###end play
        attn_output, _ = self.multihead_attn(query, key, value,attn_mask=attn_mask)  
        attn_output = attn_output.permute(1, 0, 2)  

        # Step 3: Add and norm
        attn_output = self.dropout1(attn_output)  
        attn_output = self.norm1(attn_output + depth_feat)

        # Step 4: Apply Feed Forward Network (FFN)
        ffn_output = self.ffn(attn_output)  

        # Step 5: Add and norm
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.norm2(ffn_output + attn_output)

        return ffn_output
