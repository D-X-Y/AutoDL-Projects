##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np

if sys.version_info.major == 2:  # Python 2.x
    from StringIO import StringIO as BIO
else:  # Python 3.x
    from io import BytesIO as BIO

if importlib.util.find_spec("tensorflow"):
    import tensorflow as tf


class PrintLogger(object):
    def __init__(self):
        """Create a summary writer logging to log_dir."""
        self.name = "PrintLogger"

    def log(self, string):
        print(string)

    def close(self):
        print("-" * 30 + " close printer " + "-" * 30)


class Logger(object):
    def __init__(self, log_dir, seed, create_model_dir=True, use_tf=False):
        """Create a summary writer logging to log_dir."""
        self.seed = int(seed)
        self.log_dir = Path(log_dir)
        self.model_dir = Path(log_dir) / "checkpoint"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        # self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        self.use_tf = bool(use_tf)
        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )
        # self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime( '%d-%h-at-%H:%M:%S', time.gmtime(time.time()) )))
        self.logger_path = self.log_dir / "seed-{:}-T-{:}.log".format(
            self.seed, time.strftime("%d-%h-at-%H-%M-%S", time.gmtime(time.time()))
        )
        self.logger_file = open(self.logger_path, "w")

        if self.use_tf:
            self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
            self.writer = tf.summary.FileWriter(str(self.tensorboard_dir))
        else:
            self.writer = None

    def __repr__(self):
        return "{name}(dir={log_dir}, use-tf={use_tf}, writer={writer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def path(self, mode):
        valids = ("model", "best", "info", "log", None)
        if mode is None:
            return self.log_dir
        elif mode == "model":
            return self.model_dir / "seed-{:}-basic.pth".format(self.seed)
        elif mode == "best":
            return self.model_dir / "seed-{:}-best.pth".format(self.seed)
        elif mode == "info":
            return self.log_dir / "seed-{:}-last-info.pth".format(self.seed)
        elif mode == "log":
            return self.log_dir
        else:
            raise TypeError("Unknow mode = {:}, valid modes = {:}".format(mode, valids))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write("{:}\n".format(string))
            self.logger_file.flush()

    def scalar_summary(self, tags, values, step):
        """Log a scalar variable."""
        if not self.use_tf:
            warnings.warn("Do set use-tensorflow installed but call scalar_summary")
        else:
            assert isinstance(tags, list) == isinstance(
                values, list
            ), "Type : {:} vs {:}".format(type(tags), type(values))
            if not isinstance(tags, list):
                tags, values = [tags], [values]
            for tag, value in zip(tags, values):
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=value)]
                )
                self.writer.add_summary(summary, step)
                self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        import scipy

        if not self.use_tf:
            warnings.warn("Do set use-tensorflow installed but call scalar_summary")
            return

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )
            # Create a Summary value
            img_summaries.append(
                tf.Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
            )

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        if not self.use_tf:
            raise ValueError("Do not have tensorflow")
        import tensorflow as tf

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
