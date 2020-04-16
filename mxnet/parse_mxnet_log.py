from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

filepath = 'MT-MM_MAX-EPOCH-5_output.txt'
output_dir = Path("saved/MT-MM-MXNet")
train_subdir, test_subdir = "train", "test"

train_writer = SummaryWriter(log_dir=output_dir/train_subdir)
test_writer = SummaryWriter(log_dir=output_dir/test_subdir)

with open(filepath) as fp:
    line = fp.readline()
    n_iter = 0
    while n_iter < 37600:
        if line.startswith("Train - "):
            entries = line.split(" - ")[1].split(", ")
            scalars = {
                "Accuracy/Source": float(entries[3].split(" ")[1]),
                "Accuracy/Target": float(entries[5].split(" ")[1]),
                "XEntLoss/Source": float(entries[2].split(" ")[1]),
                "XEntLoss/Target": float(entries[4].split(" ")[1]),
                "dSNELoss/Source": float(entries[6].split(" ")[1]),
                "dSNELoss/Target": float(entries[7].split(" ")[1]),
                "TotalLoss/Source": float(entries[8].split(" ")[1]),
                "TotalLoss/Target": float(entries[9].split(" ")[1]),
            }
            writer = train_writer
        elif line.startswith("Test  - "):
            entries = line.split(" - ")[1].split(", ")
            scalars = {
                "Accuracy/Source": float(entries[2].split(" ")[1])/100,
            }
            writer = test_writer
        else:
            line = fp.readline()
            continue

        n_iter = int(entries[1].split(" ")[1])
        for tag, scalar_value in scalars.items():
            writer.add_scalar(tag, scalar_value, n_iter)
        line = fp.readline()
