import itertools
import os.path
import subprocess

# List of different arguments to try
networks = ['odenet']
batch_sizes = [128]
data_percentages = [1.0]

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

venv_path = os.path.join(parent_dir, 'myenv', 'bin', 'activate')

script_filenames = ['odenet_cifar10_1.py']

adjoints = [True]

for script_filename in script_filenames:
    for network, batch_size, data_percentage in itertools.product(networks, batch_sizes, data_percentages):
        for adjoint in adjoints:
            cmd = [
                'source', venv_path,
                '&&',  # To chain multiple commands
                'python', script_filename,
                '--network', network,
                '--adjoint', str(adjoint),
                '--batch_size', str(batch_size),
                '--data_percentage', str(data_percentage)
            ]

            # Run the command using subprocess
            subprocess.run(' '.join(cmd), shell=True)
