from advertorch.attacks import FGSM
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L1PGDAttack
from advertorch.attacks import L2PGDAttack
from advertorch.attacks import LinfMomentumIterativeAttack


### 对抗攻击：FGSM攻击算法 (eps = 0.5/255, 2/255, 8/255)
adversary = FGSM(
   model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)

### 对抗攻击：PGD攻击算法 (eps = 0.5/255, 2/255, 8/255)
adversary = LinfPGDAttack(
   model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)


### 对抗攻击：L1PGD攻击算法 (eps = 100, 400, 1600)
adversary = L1PGDAttack(
   model_su, eps=1600, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)


### 对抗攻击：L2PGD攻击算法 (eps = 0.5, 2, 8)
adversary = L2PGDAttack(
   model_su, eps=8, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)


### 对抗攻击：LinfMomentumIterativeAttack攻击算法 (eps = 0.5/255, 2/255, 8/255)
adversary = LinfMomentumIterativeAttack(
   model_su, eps=8/255, eps_iter=2/255, nb_iter=80,
   rand_init=True, targeted=False)

