"""Isolated LP loss; does not change the original model files."""
from contextvars import ContextVar
import torch
import torch.nn.functional as F
ACTIVE_WEIGHTS=ContextVar('lp_weights',default=None)

class PairWeightedCE(torch.nn.CrossEntropyLoss):
 def forward(self,logits,labels):
  weights=ACTIVE_WEIGHTS.get()
  if weights is None:return super().forward(logits,labels)
  b=weights.numel();assert labels.numel()%b==0
  labels2=labels.reshape(b,-1);counts=(labels2!=self.ignore_index).sum(1)
  assert (counts>0).all() and (counts==counts[0]).all(), 'Unequal answer-token counts: cannot compare pair-mean and original token-mean loss.'
  if torch.all(weights==1):return super().forward(logits,labels)
  token_loss=F.cross_entropy(logits,labels,ignore_index=self.ignore_index,reduction='none').reshape(b,-1)
  per_pair=token_loss.float().sum(1)/counts
  return (per_pair*weights.to(per_pair.device)).mean()
