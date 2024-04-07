import torch
import torch.nn as nn
import pandas as pd
import numpy as np
np.random.seed(0)
torch.manual_seed(71)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(0.70)

    def forward(self, input):
        return 1 / (1 + torch.exp(-self.weight * input))

class FixedSigmoid(nn.Module):
    def __init__(self, ):
        super(FixedSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.reset_parameters()

    # ini_K
    def reset_parameters(self):
        self.weight.data.fill_(0.70)

    def forward(self, input):
        return 1 / (1 + torch.exp(-self.weight * input))



class hydruNN(nn.Module):
    def __init__(self):
        super(hydruNN, self).__init__()
        # self.a1 = soft_exponential(1, 0.15)

        self.lsig1 = FixedSigmoid()
        self.lsig2= LearnableSigmoid()
        self.theta_s = nn.Parameter(torch.tensor(0.50))
        self.K_s = nn.Parameter(torch.tensor(10.8))
        self.theta_s.requires_grad = False
        self.K_s.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=1, bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=1, bias=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=60, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=1, bias=True),
        )

    def forward(self, x):
        psi_log = self.fc1(x)
        psi = -torch.exp(psi_log)
        k_pre = self.fc2(-psi_log)
        K = self.K_s * torch.pow(self.lsig1(k_pre),4.0)
        theta_pre = self.fc3(-psi_log)
        theta = self.theta_s * self.lsig2(theta_pre)

        return psi, psi_log, theta, K


    def hcf_wrc(self, x):
        k_pre = self.fc2(x)
        K_ = self.K_s * torch.pow(self.lsig1(k_pre),4.0)
        theta_pre = self.fc3(x)
        theta_ =  self.theta_s * self.lsig2(theta_pre)
        return theta_, K_, theta_pre, k_pre, self.lsig1.weight, self.K_s, self.lsig2.weight, self.theta_s

  # def forward(self, x):
    #     psi_log = self.fc1(x)
    #     psi = -torch.exp(psi_log)
    #     k_pre = self.fc2(-psi_log)
    #     K = self.K_s * self.lsig(k_pre)
    #     theta_pre = self.fc3(-psi_log)
    #     theta = self.theta_s * torch.sigmoid(theta_pre)
    #
    #     return psi, psi_log, theta, K

    # def forward(self, x):
    #     psi_log = self.fc1(x)
    #     psi = -torch.exp(psi_log)
    #     k_pre = self.fc2(-psi_log)
    #     K = self.K_s * torch.pow(self.lsig1(k_pre),2)
    #     theta_pre = self.fc3(-psi_log)
    #     theta = self.theta_s * self.lsig2(theta_pre)
    #
    #     return psi, psi_log, theta, K




class PhysicsInformedNN:
    def __init__(self, tt, zt, theta, tp, zp, psi, tf, zf):
        self.tt = torch.tensor(tt, requires_grad=True).float().to(device)
        self.zt = torch.tensor(zt, requires_grad=True).float().to(device)
        self.tp = torch.tensor(tp, requires_grad=True).float().to(device)
        self.zp = torch.tensor(zp, requires_grad=True).float().to(device)
        self.theta = torch.tensor(theta).float().to(device)
        self.psi = torch.tensor(psi).float().to(device)
        self.tf = torch.tensor(tf, requires_grad=True).float().to(device)
        self.zf = torch.tensor(zf, requires_grad=True).float().to(device)

        # deep neural networks
        self.dnn = hydruNN().to(device)
        self.dnn.apply(self.weight_ini_x)

        self.lambda_1 = torch.tensor([0.005], requires_grad=False).to(device)
        self.lambda_2 = torch.tensor([0.000000035], requires_grad=False).to(device)
        self.lambda_3 = torch.tensor([1.0], requires_grad=False).to(device)
        # self.dnn.fc2.apply(self.weight_ini_K)

        # self.constraints = weightConstraint()
        # self.dnn.fc2.apply(self.constraints)
        # self.dnn.fc3.apply(self.constraints)

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1,
            max_iter=100000,
            max_eval=100000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.iter = 0

    def weight_ini(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight.data, 0., 0.1)
                nn.init.constant_(m.bias.data, 0)

    def weight_ini_x(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data = m.weight.data * m.weight.data.float()
                nn.init.constant_(m.bias.data, 0)
                print(m.weight)
                print(m.bias)

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        psi, psi_log, theta, K = self.net_u(x, t)
        psi_z = torch.autograd.grad(
            psi, x,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True
        )[0]
        psi_zz = torch.autograd.grad(
            psi_z, x,
            grad_outputs=torch.ones_like(psi_z),
            retain_graph=True,
            create_graph=True
        )[0]
        theta_t = torch.autograd.grad(
            theta, t,
            grad_outputs=torch.ones_like(theta),
            retain_graph=True,
            create_graph=True
        )[0]
        K_z = torch.autograd.grad(
            K, x,
            grad_outputs=torch.ones_like(K),
            retain_graph=True,
            create_graph=True
        )[0]
        f = theta_t - K_z * psi_z - K_z - K * psi_zz

        return f

    def loss_func(self):
        theta_psi, theta_psi_log, theta_theta, theta_K = self.net_u(self.zt, self.tt)
        psi_psi, psi_psi_log, psi_theta, psi_K = self.net_u(self.zp, self.tp)
        f_pred = self.net_f(self.zf, self.tf)
        theta_loss = torch.sum((self.theta - theta_theta) ** 2.)
        psi_loss = torch.sum((self.psi - psi_psi_log) ** 2.)
        f_loss = torch.sum(f_pred ** 2.)
        loss = self.lambda_1*theta_loss + self.lambda_2 * psi_loss + self.lambda_3 * f_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.iter += 1
        if self.iter % 10 == 0:
            print(
                'iteration： %d，Loss: %e, Loss1: %e, Loss2: %e, Loss3: %e' %
                (
                    self.iter,
                    loss.item(),
                    theta_loss.item(),
                    psi_loss.item(),
                    f_loss.item()
                )
            )
        return loss

    def train(self, nIter):
        self.dnn.train()
        # self.dnn._modules['fc2'].apply(self.constraints)
        for epoch in range(nIter):
            theta_psi, theta_psi_log, theta_theta, theta_K = self.net_u(self.zt, self.tt)
            psi_psi, psi_psi_log, psi_theta, psi_K = self.net_u(self.zp, self.tp)
            f_pred = self.net_f(self.zf, self.tf)
            theta_loss = torch.sum((self.theta - theta_theta) ** 2.)
            psi_loss = torch.sum((self.psi - psi_psi_log) ** 2.)
            f_loss = torch.sum(f_pred ** 2.)
            loss = self.lambda_1*theta_loss + self.lambda_2 * psi_loss + self.lambda_3 * f_loss
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            # self.dnn._modules['fc2'].apply(self.constraints)

            if epoch % 100 == 0:
                #Lm1: %.3e, Lm2: %.3e, Lm3: %.3e
                print(
                    'It: %d, Loss: %.3e,Loss1: %.3e, Loss2: %.3e, Loss3: %.3e' %
                    (
                        epoch,
                        loss.item(),
                        theta_loss.item(),
                        psi_loss.item(),
                        f_loss.item()
                    )
                )
        print('lbfgs_start')
        self.optimizer.step(self.loss_func)
        self.dnn.eval()
        torch.save(self.dnn.state_dict(), 'E:\\PINN\\标准试验800-20\\silt_loam\\result\\stdexp_result747.pth')


    def predict(self, z, t):
        x = torch.tensor(z, requires_grad=True).float().to(device)
        t = torch.tensor(t, requires_grad=True).float().to(device)

        self.dnn.eval()
        psi, psi_log, theta, K = self.net_u(x, t)
        # print("psi", psi.size())
        # print("theta", psi.size())
        # print("K", psi.size())
        # print(K)

        psi_z = torch.autograd.grad(
            psi, x,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,
            create_graph=True
        )[0]
        # print("psi_z", psi_z.size(), )
        # psi_z_s = psi_z.squeeze()
        # print("psi_z_s", psi_z_s.size())
        psi_zz = torch.autograd.grad(
            psi_z, x,
            grad_outputs=torch.ones_like(psi_z),
            retain_graph=True,
            create_graph=True
        )[0]
        # print("psi_zz", psi_zz.size(), psi_zz)
        theta_t = torch.autograd.grad(
            theta, t,
            grad_outputs=torch.ones_like(theta),
            retain_graph=True,
            create_graph=True
        )[0]
        # print("theta_t", theta_t.size())
        K_z = torch.autograd.grad(
            K, x,
            grad_outputs=torch.ones_like(K),
            retain_graph=True,
            create_graph=True
        )[0]

        f = theta_t - K_z * psi_z - K_z -K*psi_zz
        # print("f", f.size())
        flux = - K * (psi_z + 1.0)

        return theta.detach().cpu().numpy(), psi.detach().cpu().numpy(), psi_log.detach().cpu().numpy(), K.detach().cpu().numpy(), f.detach().cpu().numpy(), theta_t.detach().cpu().numpy(), \
               psi_z.detach().cpu().numpy(), psi_zz.detach().cpu().numpy(), K_z.detach().cpu().numpy(), flux.detach().cpu().numpy()

    def HCF_WRC(self, psi_log):
        self.dnn.eval()
        psi_log = torch.tensor(psi_log).float().to(device)
        theta_look, K_look, theta_pre, k_pre, lsig1_weight, K_s, lsig2_weight, theta_s = self.dnn.hcf_wrc(psi_log)
        return theta_look.detach().cpu().numpy(), K_look.detach().cpu().numpy(), theta_pre.detach().cpu().numpy(), k_pre.detach().cpu().numpy(),lsig1_weight.detach().cpu().numpy(),\
               K_s.detach().cpu().numpy(), lsig2_weight.detach().cpu().numpy(), theta_s.detach().cpu().numpy(), self.lambda_2.detach().cpu().numpy()


# data_psi = pd.read_csv("E:\\PINN\\data\\new\\data_psi_nod_info.csv", float_precision='round_trip')
# data_psi = pd.read_csv("E:\\PINN\\data\\loam\\row\\data_psi_nod_info_6.csv", float_precision='round_trip')
# data_psi = pd.read_csv("E:\\PINN\\标准试验800-20\\loam\\training\\data_psi_nod_info.csv", float_precision='round_trip')
data_psi = pd.read_csv("E:\\PINN\\标准试验800-20\\silt_loam\\training\\data_psi_nod_info.csv", float_precision='round_trip')
# data_psi = pd.read_csv("E:\\PINN\\标准试验800-20\\sandy_loam\\training\\data_psi_nod_info.csv", float_precision='round_trip')

# data_theta = pd.read_csv("E:\\PINN\\data\\new\\data_theta_nod_info.csv", float_precision='round_trip')
# data_theta = pd.read_csv("E:\\PINN\\data\\loam\\row\\data_theta_nod_info_6.csv", float_precision='round_trip')
# data_theta = pd.read_csv("E:\\PINN\\标准试验800-20\\loam\\training\\data_theta_nod_info.csv", float_precision='round_trip')
data_theta = pd.read_csv("E:\\PINN\\标准试验800-20\\silt_loam\\training\\data_theta_nod_info.csv", float_precision='round_trip')
# data_theta = pd.read_csv("E:\\PINN\\标准试验800-20\\sandy_loam\\training\\data_theta_nod_info.csv", float_precision='round_trip')

tt = data_theta['time'].values[:,None]
zt = data_theta['z'].values[:,None].astype(np.float32)
theta = data_theta['theta'].values[:,None]
tp = data_psi['time'].values[:,None]
zp = data_psi['z'].values[:,None].astype(np.float32)
psi = data_psi['psi'].values[:,None]
noise = 0.005
noise_theta = noise*np.random.randn(theta.shape[0], theta.shape[1])
theta_noise = theta + noise_theta
noise_psi = psi*noise*np.random.randn(psi.shape[0], psi.shape[1])
psi_noise = psi + noise_psi

#物理信息输入节点
z = np.arange(0.0, -100.0, -1)
t = np.arange(25.0, 0.0, -0.1)
np.random.shuffle(z)
np.random.shuffle(t)
xv, yv = np.meshgrid(z, t)
zf = xv.reshape(250*100, 1)
tf = yv.reshape(250*100, 1)



model = PhysicsInformedNN(tt, zt, theta, tp, zp, psi, tf, zf)

model.train(1)

dataset = pd.DataFrame({'zt': zt.flatten(),
                        'tt': tt.flatten(),
                        'theta_actual': theta_noise.flatten(),
                        'zp': zp.flatten(),
                        'tp': tp.flatten(),
                        'psi_log': psi_noise.flatten(),
                        'psi_act': -np.exp(psi).flatten()
                        })
dataset.to_csv("E:\\PINN\\标准试验800-20\\loam\\result\\datal1_val.csv")


dataset = pd.DataFrame({'zf': zf.flatten(),
                        'tf': tf.flatten()
                        })
dataset.to_csv("E:\\PINN\\标准试验800-20\\loam\\result\\datal2_val.csv")


theta_star, psi_star, psi_log_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star = model.predict(zp,tp)
dataset = pd.DataFrame({'z': zp.flatten(),
                        't': tp.flatten(),
                    'theta_pred': theta_star.flatten(),
                    'psi_act':psi.flatten(),
                    'psi_pred':psi_star.flatten(),
                    'K_pred': K_star.flatten(),
                    'flux_pred': flux_star.flatten(),
                    'f_pred': f_star.flatten(),
                    'theta_t_pred': theta_t_star.flatten(),
                    'psi_z_pred': psi_z_star.flatten(),
                    'psi_zz_pred': psi_zz_star.flatten(),
                    'K_z_pred': K_z_star.flatten()})
dataset.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_psi.csv")


theta_star, psi_star, psi_log_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star = model.predict(zt,tt)
dataset = pd.DataFrame({'z': zt.flatten(),
                        't': tt.flatten(),
                    'theta_actual': theta.flatten(),
                    'theta_pred': theta_star.flatten(),
                    'psi_pred':psi_star.flatten(),
                    'K_pred': K_star.flatten(),
                    'flux_pred': flux_star.flatten(),
                    'f_pred': f_star.flatten(),
                    'theta_t_pred': theta_t_star.flatten(),
                    'psi_z_pred': psi_z_star.flatten(),
                    'psi_zz_pred': psi_zz_star.flatten(),
                    'K_z_pred': K_z_star.flatten()})
dataset.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_theta.csv")

log_h_look = np.arange(0.5, 7.5, 0.0125)
h_look = - np.exp(log_h_look)
psi_look = log_h_look.reshape(560, 1)
psi_look = - psi_look
theta_look, K_look, theta_pre, k_pre, lsig1_weight, K_s, lsig2_weight, theta_s, lambda_2 = model.HCF_WRC(psi_look)
lookup = pd.DataFrame({'psi': h_look.flatten(),
                       'psi_log': psi_look.flatten(),
                       'theta': theta_look.flatten(),
                       'K': K_look.flatten(),
                       'thetapre':theta_pre.flatten(),
                       'k_pre':k_pre.flatten()
                       })
print('lsig1_weight, K_s, lsig2_weight, theta_s, lanbda2',lsig1_weight, K_s, lsig2_weight, theta_s, lambda_2)
lookup.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_lookup_wrc.csv")

log_h_look = np.arange(-10, 11, 0.0125)
h_look = - np.exp(log_h_look)
psi_look = log_h_look.reshape(1680, 1)
psi_look = - psi_look
theta_look, K_look, theta_pre, k_pre, lsig1_weight, K_s, lsig2_weight, theta_s, lambda2 = model.HCF_WRC(psi_look)
lookup = pd.DataFrame({'psi': h_look.flatten(),
                       'psi_log': psi_look.flatten(),
                       'theta': theta_look.flatten(),
                       'K': K_look.flatten(),
                       'thetapre':theta_pre.flatten(),
                       'k_pre':k_pre.flatten()
                       })
lookup.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_lookup_hcf.csv")

z = np.arange(0.0, 101.0, 1)
z = -z
t = np.arange(0.0, 100.4, 0.4)
xv, yv = np.meshgrid(z, t)
zf = xv.reshape(101*251, 1)
tf = yv.reshape(101*251, 1)

theta_star, psi_star, psi_log_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star = model.predict(zf, tf)
dataset = pd.DataFrame({'z': zf.flatten(),
                        't': tf.flatten(),
                    'theta_pred': theta_star.flatten(),
                    'psi_pred':psi_star.flatten(),
                    'K_pred': K_star.flatten(),
                    'flux_pred': flux_star.flatten()})
dataset.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_node.csv")


z = np.arange(-100.0, 1, 1)
t = np.arange(0.0, 100.4, 0.4)
yv, xv = np.meshgrid(t, z)
zf = xv.reshape(101*251, 1)
tf = yv.reshape(101*251, 1)

theta_star, psi_star, psi_log_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star = model.predict(zf, tf)
dataset = pd.DataFrame({'z': zf.flatten(),
                        't': tf.flatten(),
                    'theta_pred': theta_star.flatten(),
                    'psi_pred':psi_star.flatten(),
                    'K_pred': K_star.flatten(),
                    'flux_pred': flux_star.flatten()})
dataset.to_csv("E:\\PINN\\标准试验800-20\\silt_loam\\result\\mesh_result747_flux.csv")

# K = np.arange(-6.0, -1, 0.01)
# K_look = - 10.0^(K)