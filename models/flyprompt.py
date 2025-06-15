import logging

from models.l2p import L2P

logger = logging.getLogger()


class FlyPrompt(L2P):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)