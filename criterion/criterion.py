import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
	def __init__(self, weight=None):
		super().__init__()
		self.loss = nn.NLLLoss2d(weight)

	def forward(self, outputs, targets):
        	#torch version >0.2 F.log_softmax(input, dim=?) 
        	#dim (int): A dimension along which log_softmax will be computed.
		try:
			return self.loss(F.log_softmax(outputs,dim=1), targets) # if torch version >=0.3
		except TypeError as t:
			return self.loss(F.log_softmax(outputs), targets)       #else

