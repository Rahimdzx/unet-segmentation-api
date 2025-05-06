import torch, torch.nn as nn, torch.nn.functional as F

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,out_ch,3,1,1,bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch,out_ch,3,1,1,bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self,in_chs):
        super().__init__()
        self.blocks = nn.ModuleList([conv_block(c1,c2) for c1,c2 in zip(in_chs[:-1],in_chs[1:])])
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        feats=[]
        for blk in self.blocks:
            x=blk(x)
            feats.append(x)
            x=self.pool(x)
        return x,feats

class Decoder(nn.Module):
    def __init__(self,chs):
        super().__init__()
        self.up_convs=nn.ModuleList([nn.ConvTranspose2d(c1,c2,2,2) for c1,c2 in zip(chs[:-1],chs[1:])])
        self.blocks=nn.ModuleList([conv_block(c1,c2) for c1,c2 in zip([c1+c2 for c1,c2 in zip(chs[1:],chs[1:])],chs[1:])])
    def forward(self,x,feats):
        for up,blk,skip in zip(self.up_convs,self.blocks,reversed(feats)):
            x=up(x)
            diffY=skip.size()[2]-x.size()[2]
            diffX=skip.size()[3]-x.size()[3]
            x=F.pad(x,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
            x=torch.cat([skip,x],1)
            x=blk(x)
        return x

class UNet(nn.Module):
    def __init__(self,n_classes=1,base_c=64):
        super().__init__()
        enc_chs=[3,base_c,base_c*2,base_c*4,base_c*8]
        dec_chs=[base_c*16,base_c*8,base_c*4,base_c*2,base_c]
        self.encoder=Encoder(enc_chs)
        self.bottleneck=conv_block(enc_chs[-1],enc_chs[-1]*2)
        self.decoder=Decoder(dec_chs)
        self.out_conv=nn.Conv2d(base_c,n_classes,1)
    def forward(self,x):
        x,feats=self.encoder(x)
        x=self.bottleneck(x)
        x=self.decoder(x,feats)
        return self.out_conv(x)
