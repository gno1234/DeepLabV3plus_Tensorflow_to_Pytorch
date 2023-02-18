class ConvBlock(nn.Module):
  def __init__(self,num_filters=256, kernel_size=3, dilation_rate=1, padding_mode='same', use_bias =False):
    super().__init__()

    self.LazyConv2d = nn.LazyConv2d(out_channels=num_filters, kernel_size=kernel_size, dilation = dilation_rate, bias=use_bias, padding=padding_mode)
    self.LazyBatchNorm2d = nn.LazyBatchNorm2d()
    self.relu = nn.ReLU()

  def forward(self, x):

    x = self.LazyConv2d(x)
    x = self.LazyBatchNorm2d(x)
    x = self.relu(x)

    return x

class DSPP(nn.Module):
  def __init__(self):
    super().__init__()

    self.average_pooling_2d = nn.AvgPool2d(kernel_size=32)
    self.conv_1 = ConvBlock(kernel_size=1, use_bias=True)
    self.upsample2d_1 = nn.UpsamplingBilinear2d(size=None, scale_factor=32) 

    self.conv_2_1 = ConvBlock(kernel_size=3, dilation_rate=1) 
    self.conv_2_2 = ConvBlock(kernel_size=3, dilation_rate=6) 
    self.conv_2_3 = ConvBlock(kernel_size=3, dilation_rate=12) 
    self.conv_2_4 = ConvBlock(kernel_size=3, dilation_rate=18)

    self.conv_3 = ConvBlock(kernel_size=1)

  def forward(self, dssp_input):

    x = self.average_pooling_2d(dssp_input)
    x = self.conv_1(x)

    out_pool = self.upsample2d_1(x)

    out_1 = self.conv_2_1(dssp_input)
    out_6 = self.conv_2_2(dssp_input)
    out_12 = self.conv_2_3(dssp_input)
    out_18 = self.conv_2_4(dssp_input)

    x = torch.cat((out_pool, out_1, out_6, out_12, out_18), dim = 1)

    dssp_output = self.conv_3(x)

    return dssp_output

class test(nn.Module):
  def __init__(self):#
    super().__init__()

    self.num_classes = 2

    #resnet50_model = resnet50(pretrained=True)
    resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.resnet50_part1 = nn.Sequential(*list(resnet50_model.children())[0:5])
    self.resnet50_part2 = nn.Sequential(*list(resnet50_model.children())[5:7])

    self.DSPP = DSPP()

    self.conv_4 = ConvBlock(num_filters=48, kernel_size=1)
    self.conv_5 = ConvBlock()
    self.conv_6 = ConvBlock()
    self.upsample2d_2 = nn.UpsamplingBilinear2d(size=None, scale_factor=4)
    self.upsample2d_3 = nn.UpsamplingBilinear2d(size=None, scale_factor=4)
    self.conv2d = torch.nn.Conv2d(in_channels = 256, out_channels = self.num_classes, kernel_size =1, padding_mode='zeros')

  def forward(self, x):

    input_b = self.resnet50_part1(x)
    dspp_input = self.resnet50_part2(input_b)

    dspp_output = self.DSPP(dspp_input)

    input_a = self.upsample2d_2(dspp_output)
    input_b = self.conv_4(input_b)

    x = torch.cat((input_a,input_b), dim = 1)

    x = self.conv_5(x)
    x = self.conv_6(x)

    x = self.upsample2d_3(x)

    x = self.conv2d(x)

    return x