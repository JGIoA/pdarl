from typing import List, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _use_grad_for_differentiable,
    _get_scalar_dtype,
)

__all__ = ["ACFGM"]


class ACFGM(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        beta: float = 0.1,
        eps: float = 1e-8,
        lims: List[float] = [-1, 1],
        *,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # Corollary 2 beta \in (0, 1 - 3^0.5 / 2]
        # if not 0.0 <= beta < 0.134:
        if not 0.0 <= beta < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {beta}")

        defaults = dict(
            beta=beta,
            eps=eps,
            lims=lims,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    p_state["step"] = torch.tensor(
                        float(p_state["step"]), dtype=_get_scalar_dtype(is_fused=fused)
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        old_grads,
        old_old_grads,
        old_old_params,
        old_old_objs,
        old_ys,
        old_taus,
        old_old_taus,
        old_etas,
        L0s,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                old_grads.append(p.grad)
                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    self.device = p.device
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["old_old_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    state["old_old_param"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    state["old_old_obj"] = torch.zeros([p.shape[0], 1], device=p.device)

                    state["old_y"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).copy_(p)
                    state["old_tau"] = (
                        torch.ones(
                            [p.shape[0], 1], dtype=_get_scalar_dtype(), device=p.device
                        )
                        * -1.0
                    )
                    state["old_old_tau"] = (
                        torch.ones(
                            [p.shape[0], 1], dtype=_get_scalar_dtype(), device=p.device
                        )
                        * -1.0
                    )
                    state["old_eta"] = (
                        torch.ones(
                            [p.shape[0], 1], dtype=_get_scalar_dtype(), device=p.device
                        )
                        * -1.0
                    )
                    state["L0"] = torch.tensor(
                        1.0, dtype=_get_scalar_dtype(), device=p.device
                    )

                old_old_grads.append(state["old_old_grad"])
                old_old_params.append(state["old_old_param"])
                old_old_objs.append(state["old_old_obj"])
                old_ys.append(state["old_y"])
                old_taus.append(state["old_tau"])
                old_old_taus.append(state["old_old_tau"])
                old_etas.append(state["old_eta"])
                L0s.append(state["L0"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        closure = torch.enable_grad()(closure)
        old_obj = closure()

        for group in self.param_groups:
            params_with_grad = []
            old_grads = []
            old_old_grads = []
            old_old_params = []
            old_old_objs = []
            old_ys = []
            old_taus = []
            old_old_taus = []
            old_etas = []
            L0s = []
            beta = group["beta"]
            lims = group["lims"]

            self._init_group(
                group,
                params_with_grad,
                old_grads,
                old_old_grads,
                old_old_params,
                old_old_objs,
                old_ys,
                old_taus,
                old_old_taus,
                old_etas,
                L0s,
            )

            # acfgm(params_with_grad, old_grads, old_old_grads, old_old_params,
            #       old_old_objs, old_ys, old_taus, old_old_taus, old_etas,
            #       beta, old_obj, lims, L0s, self.device)

            acfgm_noLsearch(
                params_with_grad,
                old_grads,
                old_old_grads,
                old_old_params,
                old_old_objs,
                old_ys,
                old_taus,
                old_old_taus,
                old_etas,
                beta,
                old_obj,
                lims,
                self.device,
            )

        return old_obj


def acfgm(
    params: List[Tensor],
    old_grads: List[Tensor],
    old_old_grads: List[Tensor],
    old_old_params: List[Tensor],
    old_old_objs: List[Tensor],
    old_ys: List[Tensor],
    old_taus: List[Tensor],
    old_old_taus: List[Tensor],
    old_etas: List[Tensor],
    beta: float,
    old_obj: Tensor,
    lims: List[float],
    L0s: List[Tensor],
    device: str,
):
    # step t
    for i, param in enumerate(params):
        # get x and g(x)
        old_param = param.detach().clone()  # x_t-1
        old_old_param = old_old_params[i]  # x_t-2
        old_grad = old_grads[i]  # g(x_t-1)
        old_old_grad = old_old_grads[i]  # g(x_t-2)

        # get optimizer states
        old_y = old_ys[i]  # y_t-1
        old_old_obj = old_old_objs[i]  # f_t-2
        old_tau = old_taus[i]  # tau_t-1
        old_old_tau = old_old_taus[i]  # tau_t-2
        old_eta = old_etas[i]  # eta_t-1

        # t = 1, 2
        if (old_old_tau == -1.0).any():
            L0 = L0s[i]
            # t = 1, f_t-1 = -1
            if (old_tau == -1.0).any():
                # Corollary 2: eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                # using upper lim for now
                eta = (
                    torch.ones_like(old_old_obj, device=device) * 1 / (3 * L0)
                )  # eta_1, init guess L = 1
                tau = torch.zeros_like(old_old_obj, device=device)  # tau_1

                # update optimizer state
                old_old_param.copy_(old_param)
                old_old_grad.copy_(old_grad)
                old_old_obj.copy_(old_obj)
                old_old_tau.copy_(old_tau)
                old_tau.copy_(tau)
                old_eta.copy_(eta)

                # acfgm udpate for x_t-1 -> x_t
                param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))

            # Corollary 2: t = 2, eta_2 = beta / (2 * L_1)
            else:
                # (2.5) L_1 = \|g_1 - g_0 \| / \|x_1 - x_0 \|
                old_L = (
                    (old_grad - old_old_grad)
                    .norm(2, dim=1, keepdim=True)
                    .div((old_param - old_old_param).norm(2, dim=1, keepdim=True))
                )  # L_1

                # Corollary 2: eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                eta_ulim = 1 / (3 * old_L.min())
                eta_llim = beta / (4 * (1 - beta) * old_L.max())
                if ((eta_llim <= old_eta) & (old_eta <= eta_ulim)).all():
                    eta = 1 / (2 * old_L)  # eta_2
                    tau = torch.ones_like(old_old_obj, device=device) * 2  # tau_2

                    # update optimizer state
                    old_old_param.copy_(old_param)
                    old_old_grad.copy_(old_grad)
                    old_old_obj.copy_(old_obj)
                    old_old_tau.copy_(old_tau)
                    old_tau.copy_(tau)
                    old_eta.copy_(eta)

                    # acfgm udpate for x_t-1 -> x_t
                    param.copy_(
                        update(eta, tau, old_y, old_grad, old_param, beta, lims)
                    )

                else:
                    if old_eta.max() <= eta_llim:
                        L0.div_(2)
                    else:
                        L0.mul_(2)

                    # revert to step 0
                    old_tau.copy_(torch.ones_like(old_old_obj, device=device) * -1.0)
                    old_old_tau.copy_(
                        torch.ones_like(old_old_obj, device=device) * -1.0
                    )
                    param.copy_(old_old_param)
                    old_y.copy_(old_old_param)

        # t >= 3
        else:
            # compute L_t-1
            # (eq 2.6) f_t-2 - f_t-1 - g_t-1 @ (x_t-2 - x_t-1)
            denom = (
                old_old_obj
                - old_obj
                - torch.sum(
                    old_grad.mul(old_param - old_old_param), dim=1, keepdim=True
                )
            )
            # (eq 2.6) L=t-1 = \| g_t-1 - g_t-2 \|^2 / (2 * denom) if denom > 0 , else 0
            old_L = torch.zeros_like(old_old_obj)
            old_L[denom > 0] = (
                (old_grad - old_old_grad)
                .norm(2, dim=1, keepdim=True)
                .pow(2)
                .div(2 * denom)[denom > 0]
            )

            # (eq 2.30) eta_t = min{ (tau_t-2 + 1) / tau_t-1 * eta_t-1), beta * tau_t-1 / (4 * L_t-1) }
            eta = torch.minimum(
                (old_old_tau + 1) / old_tau * old_eta, beta * old_tau * 0.25 / old_L
            )
            # (eq 2.31) tau_t = tau_t-1 + 2 * eta_t * L_t-1 / (beta * tau_t-1), with alpha = 0
            tau = old_tau + 2 * eta * old_L / (beta * old_tau)

            # update optimizer state
            old_old_param.copy_(old_param)
            old_old_grad.copy_(old_grad)
            old_old_obj.copy_(old_obj)
            old_old_tau.copy_(old_tau)
            old_tau.copy_(tau)
            old_eta.copy_(eta)

            # acfgm udpate for x_t-1 -> x_t
            param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))


def update(
    eta: Tensor,
    tau: Tensor,
    old_y: Tensor,
    old_grad: Tensor,
    old_param: Tensor,
    beta: float,
    lims: List[float],
):
    # (eq 2.2) z_t = proj_X{y_t-1 - eta_t * g(x_t-1)}
    z = torch.sub(old_y, old_grad.mul(eta)).clamp(lims[0], lims[1])
    # (eq 2.3) y_t = (1 - beta_t) * y_t-1 + beta_t * z_t, then store y_t
    old_y.lerp_(end=z, weight=beta)
    # (eq 2.4) x_t = (z_t + tau_t * x_t-1) / (1 + tau_t)
    # x_t = (1 - 1/(1 + tau_t) * x_t-1) + (1/(1 + tau_t) * z_t)
    # with gamma = 1/(1 + tau_t)
    gamma = 1 / (1 + tau)
    param = torch.lerp(old_param, z, gamma)
    return param


def acfgm_noLsearch(
    params: List[Tensor],
    old_grads: List[Tensor],
    old_old_grads: List[Tensor],
    old_old_params: List[Tensor],
    old_old_objs: List[Tensor],
    old_ys: List[Tensor],
    old_taus: List[Tensor],
    old_old_taus: List[Tensor],
    old_etas: List[Tensor],
    beta: float,
    old_obj: Tensor,
    lims: List[float],
    device: str,
):
    # step t
    for i, param in enumerate(params):
        # get x and g(x)
        old_param = param.detach().clone()  # x_t-1
        old_old_param = old_old_params[i]  # x_t-2
        old_grad = old_grads[i]  # g(x_t-1)
        old_old_grad = old_old_grads[i]  # g(x_t-2)

        # get optimizer states
        old_y = old_ys[i]  # y_t-1
        old_old_obj = old_old_objs[i]  # f_t-2
        old_tau = old_taus[i]  # tau_t-1
        old_old_tau = old_old_taus[i]  # tau_t-2
        old_eta = old_etas[i]  # eta_t-1

        # t = 1, 2
        if (old_old_tau == -1.0).any():
            # t = 1, f_t-1 = -1
            if (old_tau == -1.0).any():
                # eta_1 \in [ beta / (4 * (1-beta) * L_1), 1 / (3 * L_1) ]
                # using upper lim for now
                eta = torch.ones_like(old_old_obj, device=device) * 1 / (3 * 2)  # eta_1
                tau = torch.zeros_like(old_old_obj, device=device)  # tau_1

            # t = 2, eta_2 = beta / (2 * L_1)
            else:
                # (2.5) L_1 = \|g_1 - g_0 \| / \|x_1 - x_0 \|
                old_L = (
                    (old_grad - old_old_grad)
                    .norm(2, dim=1, keepdim=True)
                    .div((old_param - old_old_param).norm(2, dim=1, keepdim=True))
                )  # L_1
                eta = (1 / (2 * old_L)).clamp(0, 1)  # eta_2 tmp bounded for small old_L
                tau = torch.ones_like(old_old_obj, device=device) * 2  # tau_2

        # t >= 3
        else:
            # compute L_t-1
            # (eq 2.6) f_t-2 - f_t-1 - g_t-1 @ (x_t-2 - x_t-1)
            denom = (
                old_old_obj
                - old_obj
                - torch.sum(
                    old_grad.mul(old_param - old_old_param), dim=1, keepdim=True
                )
            )
            # (eq 2.6) L=t-1 = \| g_t-1 - g_t-2 \|^2 / (2 * denom) if denom > 0 , else 0
            old_L = torch.zeros_like(old_old_obj, device=device)
            old_L[denom > 0] = (
                (old_grad - old_old_grad)
                .norm(2, dim=1, keepdim=True)
                .pow(2)
                .div(2 * denom)[denom > 0]
            )

            # (eq 2.30) eta_t = min{ (tau_t-2 + 1) / tau_t-1 * eta_t-1), beta * tau_t-1 / (4 * L_t-1) }
            eta = torch.minimum(
                (old_old_tau + 1) / old_tau * old_eta, beta * old_tau * 0.25 / old_L
            )
            # (eq 2.31) tau_t = tau_t-1 + 2 * eta_t * L_t-1 / (beta * tau_t-1), with alpha = 0
            tau = old_tau + 2 * eta * old_L / (beta * old_tau)

        # update optimizer state
        old_old_param.copy_(old_param)
        old_old_grad.copy_(old_grad)
        old_old_obj.copy_(old_obj)
        old_old_tau.copy_(old_tau)
        old_tau.copy_(tau)
        old_eta.copy_(eta)

        # acfgm udpate for x_t-1 -> x_t
        param.copy_(update(eta, tau, old_y, old_grad, old_param, beta, lims))


if __name__ == "__main__":
    import torch
    from torch import nn
    from torch.autograd import Variable

    class obj(nn.Module):
        def __init__(self):
            super(obj, self).__init__()

        def forward(self, x):
            # return - x**2 # + 10*x**2
            return torch.sum(-1 * x**2 + 6 * x, dim=1, keepdim=True)

    device = 'cpu'
    # device = "cuda:0"
    net0 = obj().to(device)
    act_opt = Variable(torch.rand([2, 2], device=device), requires_grad=True)
    optimizer_act = ACFGM([act_opt], beta=0.26, lims=[-10, 10])
    v = torch.ones(act_opt.shape[0], 1, device=device)
    for i in range(50):

        def closure():
            optimizer_act.zero_grad()
            out = -net0(act_opt)
            out.backward(gradient=v, retain_graph=True)
            return out

        out = optimizer_act.step(closure)
        print(i, out.detach().squeeze(), act_opt.detach().squeeze())
