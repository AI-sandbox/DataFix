from absl import logging
import time


class TimedLogger(object):
    def __init__(self, action_name):
        self.action_name = action_name

    def __enter__(self):
        self.start = time.time()
        logging.log(logging.DEBUG, "Start '{}'".format(self.action_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        logging.log(
            logging.DEBUG,
            "'{}' executed in {} seconds".format(self.action_name, end - self.start),
        )
