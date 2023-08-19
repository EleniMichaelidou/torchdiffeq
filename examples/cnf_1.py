import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--num_samples_testing', type=int, default=128)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, _hidden_dim, _width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = _hidden_dim
        self.width = _width
        self.hyper_net = HyperNetwork(in_out_dim, _hidden_dim, _width)

    def forward(self, _t, states):
        z = states[0]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(_t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return dz_dt, dlogp_z_dt


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, _hidden_dim, _width):
        super().__init__()

        blocksize = _width * in_out_dim

        self.fc1 = nn.Linear(1, _hidden_dim)
        self.fc2 = nn.Linear(_hidden_dim, _hidden_dim)
        self.fc3 = nn.Linear(_hidden_dim, 3 * blocksize + _width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = _hidden_dim
        self.width = _width
        self.blocksize = blocksize

    def forward(self, _t):
        # predict params
        params = _t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.avg = None
        self.val = None
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(num_samples):
    _points, _ = make_moons(n_samples=num_samples, noise=0.06)
    _x = torch.tensor(_points).type(torch.float32).to(device)
    _logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return _x, _logp_diff_t1


def generate_test_samples(num_testing_samples):
    test_points, _ = make_moons(n_samples=num_testing_samples, noise=0.06)
    x_test = torch.tensor(test_points).type(torch.float32).to(device)
    _logp_diff_t1_testing = torch.zeros(num_testing_samples, 1).type(torch.float32).to(device)

    return x_test, _logp_diff_t1_testing


if __name__ == '__main__':

    device = torch.device(
        f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu'
    )

    widths = [2, 4, 8, 16, 32, 64, 128, 256]
    repetitions = 10

    t0 = 0
    t1 = 10

    sample_plots_width = []  # List to store sample plots
    log_prob_plots_width = []  # List to store log probability plots

    for width in widths:

        run_dataframes = []

        if args.viz:
            # Create a unique results directory for each run
            results_dir = f"./results_moons_{width}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        for run in range(repetitions):
            average_loss_list = []  # Add this line before the training loop
            current_loss_list = []
            test_loss_list = []

            # model
            func = CNF(in_out_dim=2, _hidden_dim=args.hidden_dim, _width=width).to(device)
            optimizer = optim.Adam(func.parameters(), lr=args.lr)
            p_z0 = torch.distributions.MultivariateNormal(
                loc=torch.tensor([0.0, 0.0]).to(device),
                covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
            )
            loss_meter = RunningAverageMeter()
            loss_meter_testing = RunningAverageMeter()

            if args.train_dir is not None:
                if not os.path.exists(args.train_dir):
                    os.makedirs(args.train_dir)
                ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                if os.path.exists(ckpt_path):
                    checkpoint = torch.load(ckpt_path)
                    func.load_state_dict(checkpoint['func_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f'Loaded ckpt from {ckpt_path}')

            try:
                for itr in range(1, args.niters + 1):
                    optimizer.zero_grad()

                    x, logp_diff_t1 = get_batch(args.num_samples)

                    z_t, logp_diff_t = odeint(
                        func,
                        (x, logp_diff_t1),
                        torch.tensor([t1, t0]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

                    logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
                    loss = -logp_x.mean(0)

                    loss.backward()
                    optimizer.step()

                    loss_meter.update(loss.item())
                    average_loss_list.append(loss_meter.avg)  # Add this line to store the loss value
                    current_loss_list.append(loss.item())

                    print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

            except KeyboardInterrupt:
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f'Stored ckpt at {ckpt_path}')
            print(f'Training complete after {itr} iters.')

            if args.viz and run == repetitions - 1:
                viz_samples = 30000
                viz_timesteps = 41
                target_sample, _ = get_batch(args.num_samples)

                with torch.no_grad():
                    # Generate evolution of samples
                    z_t0 = p_z0.sample([viz_samples]).to(device)
                    logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

                    z_t_samples, _ = odeint(
                        func,
                        (z_t0, logp_diff_t0),
                        torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    # Generate evolution of density
                    x = np.linspace(-1.5, 2.5, 100)
                    y = np.linspace(-1.0, 1.5, 100)
                    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

                    z_t1 = torch.tensor(points).type(torch.float32).to(device)
                    logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

                    z_t_density, logp_diff_t = odeint(
                        func,
                        (z_t1, logp_diff_t1),
                        torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    # TESTING
                    x_testing, logp_diff_t1_testing = generate_test_samples(args.num_samples_testing)

                    z_t_testing, logp_diff_t_testing = odeint(
                        func,
                        (x_testing, logp_diff_t1_testing),
                        torch.tensor([t1, t0]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    z_t0_testing, logp_diff_t0_testing = z_t_testing[-1], logp_diff_t_testing[-1]

                    logp_x_testing = p_z0.log_prob(z_t0_testing).to(device) - logp_diff_t0_testing.view(-1)
                    loss_testing = -logp_x_testing.mean(0)

                    test_loss_list.append(loss_testing.item())

                    t = np.linspace(t0, t1, viz_timesteps)[-1]
                    z_sample = z_t_samples[-1]
                    z_density = z_t_density[-1]
                    logp_diff = logp_diff_t[-1]
                    plt.figure(figsize=(4, 4), dpi=200)
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.05, right=0.95)  # Minimize left and right margins
                    plt.axis('off')
                    plt.margins(0, 0)
                    plt.title(f"{width}", fontsize=36)
                    plt.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                               range=[[-1.5, 2.5], [-1.0, 1.5]])

                    plt.savefig(os.path.join(results_dir, f"sample_width_{width}-{int(t * 1000):05d}.jpg"),
                                pad_inches=0.0, bbox_inches='tight')

                    sample_plots_width.append(plt.gcf())
                    plt.close()

                    plt.figure(figsize=(4, 4), dpi=200)
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.05, right=0.95)  # Minimize left and right margins
                    plt.axis('off')
                    plt.margins(0, 0)
                    plt.title(f"{width}", fontsize=36)
                    logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                    plt.tricontourf(*z_t1.detach().cpu().numpy().T,
                                    np.exp(logp.detach().cpu().numpy()), 200)

                    plt.savefig(os.path.join(results_dir, f"log_width_{width}-{int(t * 1000):05d}.jpg"),
                                pad_inches=0.0, bbox_inches='tight')
                    log_prob_plots_width.append(plt.gcf())
                    plt.close()

            run_loss_df = pd.DataFrame(
                {'Average Loss': average_loss_list, 'Current Loss': current_loss_list,
                 'Test Loss': [test_loss_list[0] if test_loss_list else 0] * len(average_loss_list)})
            run_dataframes.append(run_loss_df)  # Store the DataFrame for this run

        # Calculate the average across runs and create a new DataFrame for it
        average_loss_df = pd.concat(run_dataframes).groupby(level=0).mean()

        # Create a Pandas Excel writer using XlsxWriter engine
        with pd.ExcelWriter(os.path.join(results_dir, f'loss_data_{width}.xlsx'),
                            engine='xlsxwriter') as excel_writer:
            # Save each run DataFrame to a separate sheet
            for idx, run_df in enumerate(run_dataframes):
                run_df.to_excel(excel_writer, sheet_name=f'Run{idx + 1}', index=False)

            average_loss_df.to_excel(excel_writer, sheet_name='Average', index=False)

        print(
            f'Saved loss values at {os.path.join(results_dir, f"loss_data_{width}.xlsx")}'
        )

    hidden_dims = [2, 4, 8, 16, 32, 64, 128, 256]
    repetitions = 10

    t0 = 0
    t1 = 10

    sample_plots_hidden = []
    log_prob_plots_hidden = []

    for hidden_dim in hidden_dims:

        run_dataframes = []

        if args.viz:
            # Create a unique results directory for each run
            results_dir = f"./results_moons_hidden_{hidden_dim}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        for run in range(repetitions):
            average_loss_list = []  # Add this line before the training loop
            current_loss_list = []
            test_loss_list = []

            # model
            func = CNF(in_out_dim=2, _hidden_dim=hidden_dim, _width=args.width).to(device)
            optimizer = optim.Adam(func.parameters(), lr=args.lr)
            p_z0 = torch.distributions.MultivariateNormal(
                loc=torch.tensor([0.0, 0.0]).to(device),
                covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
            )
            loss_meter = RunningAverageMeter()

            if args.train_dir is not None:
                if not os.path.exists(args.train_dir):
                    os.makedirs(args.train_dir)
                ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                if os.path.exists(ckpt_path):
                    checkpoint = torch.load(ckpt_path)
                    func.load_state_dict(checkpoint['func_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f'Loaded ckpt from {ckpt_path}')

            try:
                for itr in range(1, args.niters + 1):
                    optimizer.zero_grad()

                    x, logp_diff_t1 = get_batch(args.num_samples)

                    z_t, logp_diff_t = odeint(
                        func,
                        (x, logp_diff_t1),
                        torch.tensor([t1, t0]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

                    logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
                    loss = -logp_x.mean(0)

                    loss.backward()
                    optimizer.step()

                    loss_meter.update(loss.item())
                    average_loss_list.append(loss_meter.avg)  # Add this line to store the loss value
                    current_loss_list.append(loss.item())

                    print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

            except KeyboardInterrupt:
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f'Stored ckpt at {ckpt_path}')
            print(f'Training complete after {itr} iters.')

            if args.viz and run == repetitions - 1:
                viz_samples = 30000
                viz_timesteps = 41
                target_sample, _ = get_batch(args.num_samples)

                with torch.no_grad():
                    # Generate evolution of samples
                    z_t0 = p_z0.sample([viz_samples]).to(device)
                    logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

                    z_t_samples, _ = odeint(
                        func,
                        (z_t0, logp_diff_t0),
                        torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    # Generate evolution of density
                    x = np.linspace(-1.5, 2.5, 100)
                    y = np.linspace(-1.0, 1.5, 100)
                    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

                    z_t1 = torch.tensor(points).type(torch.float32).to(device)
                    logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

                    z_t_density, logp_diff_t = odeint(
                        func,
                        (z_t1, logp_diff_t1),
                        torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    # TESTING
                    # TESTING
                    x_testing, logp_diff_t1_testing = generate_test_samples(args.num_samples_testing)

                    z_t_testing, logp_diff_t_testing = odeint(
                        func,
                        (x_testing, logp_diff_t1_testing),
                        torch.tensor([t1, t0]).type(torch.float32).to(device),
                        atol=1e-5,
                        rtol=1e-5,
                        method='dopri5',
                    )

                    z_t0_testing, logp_diff_t0_testing = z_t_testing[-1], logp_diff_t_testing[-1]

                    logp_x_testing = p_z0.log_prob(z_t0_testing).to(device) - logp_diff_t0_testing.view(-1)
                    loss_testing = -logp_x_testing.mean(0)

                    test_loss_list.append(loss_testing.item())

                    t = np.linspace(t0, t1, viz_timesteps)[-1]
                    z_sample = z_t_samples[-1]
                    z_density = z_t_density[-1]
                    logp_diff = logp_diff_t[-1]
                    plt.figure(figsize=(4, 4), dpi=200)
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.05, right=0.95)  # Minimize left and right margins
                    plt.axis('off')
                    plt.margins(0, 0)
                    plt.title(f"{hidden_dim}", fontsize=36)
                    plt.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                               range=[[-1.5, 2.5], [-1.0, 1.5]])

                    plt.savefig(os.path.join(results_dir, f"sample_hidden_{hidden_dim}-{int(t * 1000):05d}.jpg"),
                                pad_inches=0.0, bbox_inches='tight')
                    sample_plots_hidden.append(plt.gcf())
                    plt.close()

                    plt.figure(figsize=(4, 4), dpi=200)
                    plt.tight_layout()
                    plt.subplots_adjust(left=0.05, right=0.95)  # Minimize left and right margins
                    plt.axis('off')
                    plt.margins(0, 0)
                    plt.title(f"{hidden_dim}", fontsize=36)
                    logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                    plt.tricontourf(*z_t1.detach().cpu().numpy().T,
                                    np.exp(logp.detach().cpu().numpy()), 200)

                    plt.savefig(os.path.join(results_dir, f"log_hidden_{hidden_dim}-{int(t * 1000):05d}.jpg"),
                                pad_inches=0.0, bbox_inches='tight')
                    log_prob_plots_hidden.append(plt.gcf())
                    plt.close()

            run_loss_df = pd.DataFrame(
                {'Average Loss': average_loss_list, 'Current Loss': current_loss_list,
                 'Test Loss': [test_loss_list[0] if test_loss_list else 0] * len(average_loss_list)})
            run_dataframes.append(run_loss_df)  # Store the DataFrame for this run

        # Calculate the average across runs and create a new DataFrame for it
        average_loss_df = pd.concat(run_dataframes).groupby(level=0).mean()

        # Create a Pandas Excel writer using XlsxWriter engine
        with pd.ExcelWriter(os.path.join(results_dir, f'loss_data_hid_{hidden_dim}.xlsx'),
                            engine='xlsxwriter') as excel_writer:
            # Save each run DataFrame to a separate sheet
            for idx, run_df in enumerate(run_dataframes):
                run_df.to_excel(excel_writer, sheet_name=f'Run{idx + 1}', index=False)

            average_loss_df.to_excel(excel_writer, sheet_name='Average', index=False)

        print(
            f'Saved loss values at {os.path.join(results_dir, f"loss_data_hid_{hidden_dim}.xlsx")}'
        )
    merged_sample_width_plot = np.hstack([np.array(plot.canvas.renderer.buffer_rgba()) for plot in sample_plots_width])

    # Create a new figure for saving the merged plot
    plt.figure(figsize=(32, 8))
    plt.imshow(merged_sample_width_plot)
    plt.axis('off')
    # Save the merged plot as a PDF
    plt.savefig('merged_sample_width_plot.pdf', bbox_inches='tight', pad_inches=0)

    plt.close()

    merged_log_width_plot = np.hstack([np.array(plot.canvas.renderer.buffer_rgba()) for plot in log_prob_plots_width])

    # Create a new figure for saving the merged plot
    plt.figure(figsize=(32, 8))
    plt.imshow(merged_log_width_plot)
    plt.axis('off')
    # Save the merged plot as a PDF
    plt.savefig('merged_log_width_plot.pdf', bbox_inches='tight', pad_inches=0)

    plt.close()

    merged_sample_hidden_plot = np.hstack(
        [np.array(plot.canvas.renderer.buffer_rgba()) for plot in sample_plots_hidden])

    # Create a new figure for saving the merged plot
    plt.figure(figsize=(32, 8))
    plt.imshow(merged_sample_hidden_plot)
    plt.axis('off')
    # Save the merged plot as a PDF
    plt.savefig('merged_sample_hidden_plot.pdf', bbox_inches='tight', pad_inches=0)

    plt.close()

    merged_log_hidden_plot = np.hstack([np.array(plot.canvas.renderer.buffer_rgba()) for plot in log_prob_plots_hidden])

    # Create a new figure for saving the merged plot
    plt.figure(figsize=(32, 8))
    plt.imshow(merged_log_hidden_plot)
    plt.axis('off')
    # Save the merged plot as a PDF
    plt.savefig('merged_log_hidden_plot.pdf', bbox_inches='tight', pad_inches=0)

    plt.close()
