from dataloaders.ce_mnist_dataloader import CEMNISTDataloader
from dataloaders.pn_dataloader import PNDataloader
from model.ocrnet import OCRNet

dataloader = PNDataloader('.\\ignore\\full_dataset')
sample_w, sample_h = dataloader.__getsamplesize__()
outsize = dataloader.__getclassesnum__()
timesteps = dataloader.__gettimesteps__()

model = OCRNet(img_w=sample_w, img_h=sample_h, timesteps=timesteps, outsize=outsize)

print(dataloader.__getitem__(0))
print(model)