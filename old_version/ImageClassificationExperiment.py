import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from ImageClassification import ConvQIMIA
import torch.utils.data as data_utils
from PIL import Image
def experiment(epochs,batch_size,lr):


    dataset = tv.datasets.ImageNet("/data/archive")
    dataloader = data_utils.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    test_model = ConvQIMIA(256,256,6)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(test_model.parameters(),lr)
    for epoch in range(epochs):
        for y,x in dataloader:
            optimizer.zero_grad()
            y_hat = test_model(x.float())
            loss = loss_fun(y_hat,y)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch} \t loss - {loss}\r")
def image_to_tensor(image_path):
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(), tv.transforms.Resize([512,512])])

    demo = Image.open(image_path)
    demo_img = transform(demo)
    return demo_img
def experiment0():
    test_model = ConvQIMIA(2)
    img0 = image_to_tensor("/data/archive/train.X1/n01531178/n01531178_14.JPEG")
    img1 =  image_to_tensor("/data/archive/train.X1/n01531178/n01531178_14.JPEG")
    batch = th.stack([img0,img1])
    output = test_model(batch)

if __name__ == "__main__":
    experiment0()