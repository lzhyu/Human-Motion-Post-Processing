# import torch
# from torch import nn
# from model.sttrans import PositionalEncoding
# from torch.nn import TransformerEncoderLayer
# from model.mdm import TimestepEmbedder, EmbedAction

# class TimestepEmbedder(nn.Module):
#     def __init__(self, latent_dim, sequence_pos_encoder):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.sequence_pos_encoder = sequence_pos_encoder

#         time_embed_dim = self.latent_dim
#         self.time_embed = nn.Sequential(
#             nn.Linear(self.latent_dim, time_embed_dim),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, time_embed_dim),
#         )

#     def forward(self, timesteps):
#         return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


# class ContactPredictor(nn.Module):
#     def __init__(self, input_feature, hidden_feature = 512, node_n=24, n_frames =60, num_actions = 12):
#         super(ContactPredictor, self).__init__()
#         # predict foot joints
#         self.hidden_feature = hidden_feature

#         self.frame_embed = PositionalEncoding(self.hidden_feature, 0)
#         self.embed_action = EmbedAction(num_actions, self.hidden_feature)
#         self.embed_timestep = TimestepEmbedder(self.hidden_feature, self.frame_embed)
#         self.input_linear = nn.Linear(node_n*input_feature, hidden_feature)

#         self.backbone = nn.TransformerEncoder(TransformerEncoderLayer(d_model = hidden_feature,\
#                                                  nhead=4, activation='gelu'),
#                                                 num_layers=6)
#         self.output_linear = nn.Sequential(
#             # nn.Linear(self.hidden_feature, self.hidden_feature),
#             # nn.ReLU(),
#             # nn.Linear( self.hidden_feature, self.hidden_feature),
#             # nn.ReLU(),
#             nn.Linear(self.hidden_feature, 4),# predict contact for four joints
#         )
#     def forward(self, x,  timesteps, y=None):
#         # B, V, C, T
#         emb = self.embed_timestep(timesteps)
#         action_emb = self.embed_action(y['action'])
#         emb += action_emb

#         B, V, C, T = x.shape
#         x = x.permute(3, 0, 1, 2).reshape(T, B, -1)# To T,B,V*C
#         x = self.input_linear(x)
#         # T, B, Hidden
#         x = torch.cat((emb, x), axis=0)
#         # T+1, B, Hidden

#         x = self.frame_embed(x)
#         x = self.backbone(x)
#         x = x[1:]
#         x = self.output_linear(x)
#         x = x.reshape(T, B, 2, 2).permute(1, 2, 3, 0)
#         # T, B, 8 -> B, 2, 2, T
#         return x
    
#     def get_foot_labels(self, x, y=None):
#         # generate timesteps
#         # range from timesteps 5 to 20
#         B = x.size(0)
#         assert B==1
#         timesteps = torch.arange(5,20).to(x.device) * torch.ones(B).to(x.device)# 15
#         timesteps = timesteps.long()

#         y['action'] = y['action'].repeat(15,1)# 15, 1
#         x = x.repeat(15,1,1,1)
#         # 15, 25, 6, 60
#         contact_preds = self.forward(x, timesteps, y)
#         # 15,2,2,T
        
#         # softmax + threshold
#         contact_preds = torch.nn.functional.softmax(contact_preds, dim=1)

#         contact_preds = contact_preds.mean(dim=0, keepdim=True)
#         contact_labels = (contact_preds[:,1] > 0.2).long()# original 0.3
#         # contact_labels = contact_preds.argmax(dim=1)# B,2,T
#         contact_labels_foot = contact_labels.permute(0,2,1)
#         # B,T,2
#         return contact_labels_foot
