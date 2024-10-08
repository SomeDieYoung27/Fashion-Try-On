import torch
import torch.nn as nn
import torch.nn.funtional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch,out_ch,padding=1 * dirate,dilation= 1 * dirate
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
    
    ##upsample tensor 'src' to have the same spatial size with tensor 'tar'
    def __upsample__like(src,tar):
        src = F.upsample(src,size=tar.shape[2:],mode="bilinear")

        return src
    
    ### RSU-7 ###

    class RSU7(nn.Module):  # UNet07DRES(nn.Module):
     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)


        self.rebnconv6d = REBNCONV(mid_ch * 2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6, hx7
        del hx6d, hx5d, hx3d, hx2d
        del hx2dup, hx3dup, hx4dup, hx5dup, hx6dup
        """

        return hx1d + hxin
    
 ###RSU-6###

class RSU6(nn.Module):
   def __init__(self,in_ch=3,mid_ch=12,out_ch=3):
      super(RSU6,self).__init__()

      self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

      self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
      self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
      self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
      self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

      self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
      self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

      self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
      self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
      self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
      self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
      self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
      self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
      self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
      self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
      self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
   





