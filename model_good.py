import torch
from torch import nn
import time
import os
import numpy as np

torch.set_default_dtype(torch.float64)

use_gpu = torch.cuda.is_available()
print("Use GPU? ",use_gpu)



class mynet(nn.Module):
	def __init__(self, input_channels):
		super(mynet, self).__init__()
		self.conv1 = nn.Conv1d(input_channels, 3, 5)
		self.conv2 = nn.Conv1d(3, 2, 3)
		self.pool1 = nn.AvgPool1d(2)
		self.pool2 = nn.AvgPool1d(2)
		self.relu = nn.ReLU()
		self.dp = nn.Dropout(p=0.3)
		self.fc1 = nn.Linear(3*2, 3)
		self.fc2 = nn.Linear(16, 3)

	def forward(self,x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool1(x)
		x = self.dp(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool2(x)
		x = self.dp(x)
		x = x.flatten(1)
		x = self.fc1(x)
		return x


cost = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

model = mynet(3)
