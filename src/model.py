import torch
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Encoder Code
class Generator(nn.Module):
    def __init__(self, num_hidden, num_input_channel=3):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_input_channel, num_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_hidden, num_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_hidden * 2, num_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_hidden * 4, num_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.gen_base = nn.Sequential(
            nn.ConvTranspose2d(num_hidden * 8, num_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 4),
            nn.ReLU(True),
            # state size. (num_hidden*4) x 8 x 8
            nn.ConvTranspose2d( num_hidden * 4, num_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 2),
            nn.ReLU(True),
            # state size. (num_hidden*2) x 16 x 16
        )
        
        self.image_head = nn.Sequential(
            nn.ConvTranspose2d( num_hidden * 2, num_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(True),
            # state size. (num_hidden) x 32 x 32
            nn.ConvTranspose2d( num_hidden, num_input_channel, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.mask_head  = nn.Sequential(
            nn.ConvTranspose2d( num_hidden * 2, num_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(True),
            # state size. (num_hidden) x 32 x 32
            nn.ConvTranspose2d( num_hidden, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, image):
        enc_feature = self.encoder(image)
        noise = torch.randn_like(enc_feature)
        noised_feature = enc_feature + noise
        gen_feature = self.gen_base(noised_feature)

        modify = self.image_head(gen_feature)
        mask   = self.mask_head (gen_feature)
        new_image = mask * modify + (1-mask) * image
        return new_image, modify, mask


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, num_hidden, num_input_channel=3):
        super(Discriminator, self).__init__()
        self.base = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_input_channel, num_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden) x 32 x 32
            nn.Conv2d(num_hidden, num_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*2) x 16 x 16
            nn.Conv2d(num_hidden * 2, num_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*4) x 8 x 8
            nn.Conv2d(num_hidden * 4, num_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*8) x 4 x 4
        )

        self.acne_header = nn.Sequential(
            nn.Conv2d(num_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.quality_header = nn.Sequential(
            nn.Conv2d(num_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, image):
        self.feature = feature = self.base(image)
        acne_prob = self.acne_header(feature)
        quality_prob = self.quality_header(feature)

        acne_prob = acne_prob.view(-1)
        quality_prob = quality_prob.view(-1)

        return acne_prob, quality_prob


# Classifier Code
class Classifier(nn.Module):
    def __init__(self, num_hidden, num_input_channel=3):
        super(Classifier, self).__init__()
        self.base = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_input_channel, num_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden) x 32 x 32
            nn.Conv2d(num_hidden, num_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*2) x 16 x 16
            nn.Conv2d(num_hidden * 2, num_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*4) x 8 x 8
            nn.Conv2d(num_hidden * 4, num_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_hidden*8) x 4 x 4

            nn.Conv2d(num_hidden * 8, num_hidden * 16, 4, 2, 1, bias=False),
            # state size. (num_hidden*16) x 2 x 2
        )

        self.header = nn.Sequential(
            # nn.Conv2d(num_hidden * 16, 16, 2, 1, 0, bias=False),
            nn.Linear(num_hidden*16, 1),
            nn.Sigmoid(),
        )
        

    def forward(self, image):
        feature = self.base(image) # N, C, H, W
        self.feature = feature = feature.mean(2).mean(2)
        aprob = self.header(feature)
        aprob = torch.squeeze(aprob)

        return aprob
