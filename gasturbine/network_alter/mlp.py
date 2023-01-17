import torch
import torch.nn as nn
from torchinfo import summary
# from torch.utils.tensorboard import SummaryWriter


class MLP_pred(nn.Module):

    def __init__(self, input_size=2, action_size=10):
        super().__init__()
        middle_layer_size = action_size * 4
        middle_layer_size_2 = action_size * 2
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.l2 = nn.Linear(middle_layer_size, middle_layer_size_2)
        self.l3 = nn.Linear(middle_layer_size_2, action_size)
        self.act = nn.ELU()

    def forward(self, x):
        output = self.l1(x)
        # output = torch.sigmoid(output)
        output = self.act(output)

        output = self.l2(output)
        # output = torch.sigmoid(output)
        output = self.act(output)
        output = self.l3(output)
        return output


class MLP_error_net(nn.Module):

    def __init__(self, input_size=2, action_size=10, num_classes=1):
        super().__init__()
        middle_layer_size = action_size * 2
        self.l1 = nn.Linear(input_size, middle_layer_size)
        self.action_pad = nn.Linear(action_size, middle_layer_size)

        self.l2 = nn.Linear(middle_layer_size * 2, middle_layer_size)
        self.l3 = nn.Linear(middle_layer_size, num_classes)
        self.act = nn.ELU()

    def forward(self, x, action):
        x = self.l1(x)
        # x=torch.sigmoid(x)
        x = self.act(x)

        action = self.action_pad(action)
        # action=torch.sigmoid(action)
        action = self.act(action)
        output = torch.cat([x, action], dim=1)

        output = self.l2(output)
        # output=torch.sigmoid(output)
        output = self.act(output)

        output = self.l3(output)
        output = torch.sigmoid(output) * 2

        return output


# net = MLP_action(input_size=2, action_size=11)
# print(summary(net, (1, 2), device='cpu'))
# net = MLP_Q_net(input_size=2, action_size=11)
# print(summary(net, [(1, 2), (1, 11)], device='cpu'))
# x = torch.randn(1, 2)
# y= torch.randn(1, 11)
# vis_graph = h.build_graph(net, (x,y))   # 获取绘制图像的对象
# vis_graph.theme = h.graph.THEMES["blue"].copy()
# vis_graph.save("./demo2.png")
# writer=SummaryWriter("runs/fashion_mnist_experiment_1")

# writer.add_graph(net,(x,y))
#
# # y=net(x)
# # dot = make_dot(y.mean(), params=dict(net.named_parameters()))
# # net.to("cuda")
# writer.close()