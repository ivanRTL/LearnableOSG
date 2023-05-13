import os
from argparse import ArgumentParser

import torch
import OSG_VSD as OSG
import numpy as np

np.set_printoptions(linewidth=300)
import osg_vsd_dataset
import OptimalSequentialGrouping


def CLossTest(
    args,
    data_folder_path="h5",
    modality="visual",
    num_iters=101,
    stop_param=0.75,
    BN=True,
    DO=0.0,
    dist_metric="cosine",
    dist_type="EMBEDDING",
    learning_rate=0.005,
    weight_decay=0,
):
    if modality == "visual":
        d, K_max = 2048, 50
        feature_sizes = [d, 3000, 3000, 1000, 100]
    elif modality == "audio":
        d, K_max = 128, 50
        feature_sizes = [d, 200, 200, 100, 20]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vsd_dataset = osg_vsd_dataset.OSG_VSD_DATASET(path_to_h5=data_folder_path, device=device)

    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data = torch.utils.data.random_split(vsd_dataset, [16, 4], generator=generator1)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, collate_fn=osg_vsd_dataset.my_collate, batch_size=args.b_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, collate_fn=osg_vsd_dataset.my_collate, batch_size=1
    )

    OSG_model = OSG.OSG_C(
        feature_sizes,
        K_max=K_max,
        BN=BN,
        DO=DO,
        dist_type=dist_type,
        dist_metric=dist_metric,
        device=device,
    )

    OSG_model.to(device)

    optimizer = torch.optim.Adam(
        OSG_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    print("Starting")
    print(
        "network",
        feature_sizes,
        "dist_type",
        dist_type,
        "dist_metric",
        dist_metric,
        "stop_param",
        stop_param,
        "modality",
        modality,
    )

    first_loss = 0
    for iteration in range(num_iters):
        optimizer.zero_grad()
        all_loss = 0

        for i, a_batch in enumerate(train_dataloader):
            print(f"batch {i}")
            x, t = a_batch

            # torch.autograd.set_detect_anomaly(True)
            T_pred_new = OSG_model(x.to(device))
            loss = OSG.my_Tloss(T_pred_new.to(device), t.to(device), device=device)
            all_loss += loss.item()

            loss.backward()
            break

        if iteration == 0:
            first_loss = all_loss

        optimizer.step()

        OSG_np = OptimalSequentialGrouping.OptimalSequentialGrouping()

        F_trn = 0
        for batch in range(len(test_dataloader)):
            print(batch)
            x_orig, t_orig = batch
            t = t_orig.cpu().numpy()
            D_temp = OSG_model.DIST_FUNC(x_orig.unsqueeze(0))
            D_new = D_temp.squeeze(0).cpu().detach().numpy()
            boundaries_new = OSG_np.blockDivideDSum(D_new, t.size)
            F_temp, __, __ = OSG_np.FCO(boundaries_new, t)
            F_trn += F_temp
            print(OSG_model(x_orig))

        print(
            "Iteration "
            + str(iteration)
            + ", loss: "
            + str(all_loss)
            + ", F-score: "
            + str(F_trn)
        )

        if all_loss < stop_param * first_loss:
            break

    print("finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--b_size", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    CLossTest(args, num_iters=args.n_iters, modality="visual", data_folder_path=args.path)
