import math
import time
from functools import reduce

import torch

from src.attacks.autoattack_wrapper.autoattack.autopgd_base import APGDAttack, L1_projection
from src.attacks.autoattack_wrapper.autoattack.other_utils import L0_norm


class APGDAttackAdaptive(APGDAttack):
    def dlr_loss(self, x, y):
        rej = x.argmax(dim=-1) == x.shape[-1] - 1
        c_min = y.clone()
        c_min[rej] = x.shape[-1] - 1
        x_sorted, ind_sorted = x[:, :-1].sort(dim=1)  # remove reject class
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])
        return -(x[u, c_min] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
                1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                                                   ).detach() * self.normalize(
                t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                                                   ).detach() * self.normalize(
                t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta

        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
                                 ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
                                      ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            else:
                raise ValueError('unknowkn loss')
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        y_pred = logits.detach().max(1)[1] == y
        acc = (y_pred == y).logical_or(y_pred == logits.shape[-1] - 1)
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in [
            'L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
                [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old = n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            # print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            # print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        # [Angelo]
        # n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]
        n_fts = reduce(lambda aa, bb: aa * bb, x.shape)
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                                                              x - self.eps),
                                                    x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                                                             ) * torch.min(
                        self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                                                             ) * torch.min(
                        self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0,
                                            max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (
                                len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                            sparsegrad.sign().abs().view(x.shape[0], -1).sum(
                                dim=-1).view(
                                -1, *[1] * (len(x.shape) - 1)) + 1e-10)

                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()

                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv,
                                                                        y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv,
                                                                        y,
                                                                        self.y_target)
                    grad += grad_curr

            grad /= float(self.eot_iter)

            y_pred = logits.detach().max(1)[1]
            pred = (y_pred == y).logical_or(y_pred == logits.shape[-1] - 1)
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in [
                    'L1'] else ''
                print(
                    '[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                        i, loss_best.sum(), acc.float().mean(), str_stats))
                # print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))

            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    if self.norm in ['Linf', 'L2']:
                        fl_oscillation = self.check_oscillation(loss_steps, i,
                                                                k,
                                                                loss_best,
                                                                k3=self.thr_decr)
                        fl_reduce_no_impr = (1. - reduced_last_check) * (
                                loss_best_last_check >= loss_best).float()
                        fl_oscillation = torch.max(fl_oscillation,
                                                   fl_reduce_no_impr)
                        reduced_last_check = fl_oscillation.clone()
                        loss_best_last_check = loss_best.clone()

                        if fl_oscillation.sum() > 0:
                            ind_fl_osc = (
                                        fl_oscillation > 0).nonzero().squeeze()
                            step_size[ind_fl_osc] /= 2.0
                            n_reduced = fl_oscillation.sum()

                            x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                            grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                        k = max(k - self.size_decr, self.n_iter_min)

                    elif self.norm == 'L1':
                        sp_curr = L0_norm(x_best - x)
                        fl_redtopk = (sp_curr / sp_old) < .95
                        topk = sp_curr / n_fts / 1.5
                        step_size[fl_redtopk] = alpha * self.eps
                        step_size[~fl_redtopk] /= adasp_redstep
                        step_size.clamp_(alpha * self.eps / adasp_minstep,
                                         alpha * self.eps)
                        sp_old = sp_curr.clone()

                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0
                    # k = max(k - self.size_decr, self.n_iter_min)

        #

        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr']  # 'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            # y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()

        if self.loss != 'ce-targeted':
            acc = (y_pred == y).logical_or(y_pred == 10)

        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                  'running {}-attack with epsilon {:.5f}'.format(
                      self.norm, self.eps),
                  '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                     .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(
                iters[:-1])  # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                                                                    ) for c in
                                                                epss]),
                                                      '+'.join([str(c) for c in
                                                                iters])))

        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool,
                                                     epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                                time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best
