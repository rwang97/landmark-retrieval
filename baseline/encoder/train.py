from encoder.visualizations import Visualizations
from encoder.data_objects import LandmarkDataset, LandmarkDataLoader
from encoder.params_model import *
from encoder.model import Encoder
from utils.profiler import Profiler
from pathlib import Path
import numpy as np
import torch

def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
def get_acc(output, label):
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc

def train(run_id: str, data_dir:str, validate_data_dir:str, models_dir: Path, umap_every: int, 
          save_every: int, backup_every: int, vis_every: int, validate_every:int, force_restart: bool, 
          visdom_server: str, port: str, no_visdom: bool):
    # Create a dataset and a dataloader
    train_dataset = LandmarkDataset(data_dir, img_per_cls, train=True)
    train_loader = LandmarkDataLoader(
        train_dataset,
        cls_per_batch,
        img_per_cls,
        num_workers=6,
    )

    validate_dataset = LandmarkDataset(validate_data_dir, v_img_per_cls, train=False)
    validate_loader = LandmarkDataLoader(
        validate_dataset,
        v_cls_per_batch,
        v_img_per_cls,
        num_workers=4,
    )

    validate_iter = iter(validate_loader)
    
    criterion = torch.nn.CrossEntropyLoss()

    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    # loss_device = torch.device("cpu")
    # fixed by https://github.com/CorentinJ/Real-Time-Voice-Cloning/issues/237
    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model and the optimizer
    model = Encoder(device, loss_device)

    multi_gpu = False
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        multi_gpu = True
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_init, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25000, gamma=0.5)
    
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    pretrained_path = state_fpath

    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(pretrained_path)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, port=port, disabled=no_visdom)
    vis.log_dataset(train_dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    # Training loop
    profiler = Profiler(summarize_every=500, disabled=False)
    for step, cls_batch in enumerate(train_loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # Forward pass
        inputs = torch.from_numpy(cls_batch.data).float().to(device)
        labels = torch.from_numpy(cls_batch.labels).long().to(device)
        sync(device)
        profiler.tick("Data to %s" % device)

        embeds, output = model(inputs)
        sync(device)
        profiler.tick("Forward pass")

        loss = criterion(output, labels)
        sync(device)
        profiler.tick("Loss")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")
            
        optimizer.step()
        scheduler.step()
        profiler.tick("Parameter update")

        acc = get_acc(output, labels)
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        vis.update(loss.item(), acc, step)
        
        print("step {}, loss: {}, acc: {}".format(step, loss.item(), acc))

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            projection_dir = backup_dir / 'projections'
            projection_dir.mkdir(exist_ok=True, parents=True)
            projection_fpath = projection_dir.joinpath("%s_umap_%d.png" % (run_id, step))
            embeds = embeds.detach()
            embeds = (embeds / torch.norm(embeds, dim=1, keepdim=True)).cpu().numpy()
            vis.draw_projections(embeds, img_per_cls, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0:
            if step > 4000: # don't save until 4k steps
                print("Making a backup (step %d)" % step)

                ckpt_dir = backup_dir / 'ckpt'
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                backup_fpath = ckpt_dir.joinpath("%s_%d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)
        
        # Do validation
        if validate_every != 0 and step % validate_every == 0:
            # validation loss, acc
            model.eval()
            for i in range(num_validate):
                with torch.no_grad():
                    validate_cls_batch = next(validate_iter)
                    validate_inputs = torch.from_numpy(validate_cls_batch.data).float().to(device)
                    validat_labels = torch.from_numpy(validate_cls_batch.labels).long().to(device)
                    validate_embeds, validate_output = model(validate_inputs)
                    validate_loss = criterion(validate_output, validat_labels)
                    validate_acc = get_acc(validate_output, validat_labels)

                vis.update_validate(validate_loss.item(), validate_acc, step, num_validate)
            
            # take the last one for drawing projection
            projection_dir = backup_dir / 'v_projections'
            projection_dir.mkdir(exist_ok=True, parents=True)
            projection_fpath = projection_dir.joinpath("%s_umap_%d.png" % (run_id, step))
            validate_embeds = validate_embeds.detach()
            validate_embeds = (validate_embeds / torch.norm(validate_embeds, dim=1, keepdim=True)).cpu().numpy()
            vis.draw_projections(validate_embeds, v_img_per_cls, step, projection_fpath, is_validate=True)
            vis.save()

            model.train()

        profiler.tick("Extras (visualizations, saving)")
    