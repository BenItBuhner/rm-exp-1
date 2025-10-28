import copy


class EMAHelper:
    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.mu).add_(param.detach(), alpha=1 - self.mu)

    def ema_copy(self, module):
        copy_module = copy.deepcopy(module)
        for name, param in copy_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
        return copy_module
