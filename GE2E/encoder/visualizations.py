from encoder.data_objects import LandmarkDataset
from datetime import datetime
from time import perf_counter as timer
import matplotlib.pyplot as plt
import numpy as np
# import webbrowser
import visdom
import umap

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255 


class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", port="8870", disabled=False):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        self.validate_losses = []
        self.validate_eers = []
        print("Updating the visualizations every %d steps." % update_every)
        
        # If visdom is disabled TODO: use a better paradigm for that
        self.disabled = disabled    
        if self.disabled:
            return 
        
        # Set the environment name
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = "%s (%s)" % (env_name, now)
        
        # Connect to visdom and open the corresponding window in the browser
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True, port=port)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)
        
        # Create the windows
        self.loss_win = None
        self.eer_win = None
        # self.lr_win = None
        self.implementation_win = None
        self.projection_win = None
        self.v_projection_win = None
        self.implementation_string = ""
        
    def log_params(self):
        if self.disabled:
            return 
        from encoder import params_model
        param_string = "<b>Model parameters</b>:<br>"
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        param_string += "<b>Data parameters</b>:<br>"
        self.vis.text(param_string, opts={"title": "Parameters"})
        
    def log_dataset(self, dataset: LandmarkDataset):
        if self.disabled:
            return 
        dataset_string = ""
        dataset_string += "<b>Classes</b>: %s\n" % len(dataset.classes)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})
        
    def log_implementation(self, params):
        if self.disabled:
            return 
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string, 
            opts={"title": "Training implementation"}
        )

    def plot_loss_eer(self, losses, eers, step, name):
        self.loss_win = self.vis.line(
                [np.mean(losses)],
                [step],
                win=self.loss_win,
                update="append" if self.loss_win else None,
                name=name,
                opts=dict(
                    legend=["Train", "Validation"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss",
                )
            )

        self.eer_win = self.vis.line(
            [np.mean(eers)],
            [step],
            win=self.eer_win,
            update="append" if self.eer_win else None,
            name=name,
            opts=dict(
                legend=["Train", "Validation"],
                xlabel="Step",
                ylabel="EER",
                title="Equal error rate"
            )
        )

    def update(self, loss, eer, step):
        # Update the tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(eer)
        print("", end="")
        
        # Update the plots every <update_every> steps
        if step % self.update_every != 0:
            return
        time_string = "Step time:  mean: %5dms  std: %5dms" % \
                      (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d   Train Loss: %.4f   EER: %.4f   %s" %
              (step, np.mean(self.losses), np.mean(self.eers), time_string))

        if not self.disabled:
            self.plot_loss_eer(self.losses, self.eers, step, "Train")

            # track step time
            if self.implementation_win is not None:
                self.vis.text(
                    self.implementation_string + ("<b>%s</b>" % time_string), 
                    win=self.implementation_win,
                    opts={"title": "Training implementation"},
                )

        # Reset the tracking
        self.losses.clear()
        self.eers.clear()
        self.step_times.clear()

    def update_validate(self, loss, eer, step, num_validate):
        self.validate_losses.append(loss)
        self.validate_eers.append(eer)
        print("", end="")

        if len(self.validate_losses) < num_validate:
            return

        print("\nStep %6d   Validation Loss: %.4f   EER: %.4f" %
              (step, np.mean(self.validate_losses), np.mean(self.validate_eers)))

        if not self.disabled:
            self.plot_loss_eer(self.validate_losses, self.validate_eers, step, "Validation")

        # Reset the tracking
        self.validate_losses.clear()
        self.validate_eers.clear()

    def draw_projections(self, embeds, img_per_cls, step, out_fpath=None,
                         max_classes=20, is_validate=False):
        max_classes = min(max_classes, len(colormap))
        embeds = embeds[:max_classes * img_per_cls]
        
        n_classes = len(embeds) // img_per_cls
        ground_truth = np.repeat(np.arange(n_classes), img_per_cls)
        colors = [colormap[i] for i in ground_truth]
        
        reducer = umap.UMAP(metric="cosine")
        projected = reducer.fit_transform(embeds)
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        if is_validate:
            plt.title("Validation UMAP projection (step %d)" % step)
            if not self.disabled:
                self.v_projection_win = self.vis.matplot(plt, win=self.v_projection_win)
        else:
            plt.title("Train UMAP projection (step %d)" % step)
            if not self.disabled:
                self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()
    
    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])
        