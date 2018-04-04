import math
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
