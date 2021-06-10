#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.06 #
#####################################################
import sys
from pathlib import Path

from .time_utils import time_for_file, time_string


class Logger:
    """A logger used in xautodl."""

    def __init__(self, root_dir, prefix="", log_time=True):
        """Create a summary writer logging to log_dir."""
        self.root_dir = Path(root_dir)
        self.log_dir = self.root_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._prefix = prefix
        self._log_time = log_time
        self.logger_path = self.log_dir / "{:}{:}.log".format(
            self._prefix, time_for_file()
        )
        self._logger_file = open(self.logger_path, "w")

    @property
    def logger(self):
        return self._logger_file

    def log(self, string, save=True, stdout=False):
        string = "{:} {:}".format(time_string(), string) if self._log_time else string
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self._logger_file.write("{:}\n".format(string))
            self._logger_file.flush()

    def close(self):
        self._logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def __repr__(self):
        return "{name}(dir={log_dir}, prefix={_prefix}, log_time={_log_time})".format(
            name=self.__class__.__name__, **self.__dict__
        )
