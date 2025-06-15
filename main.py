import logging.config

from configuration import config
from methods import METHODS

logging.config.fileConfig('configuration/logging.conf')
logger = logging.getLogger()


def main():
    # Get Configurations
    args = config.base_parser()
    logger.info('Running for seeds: %s', args.seeds)
    for seed in args.seeds:
        setattr(args, 'rnd_seed', seed)
        logger.info('Configuration: %s', args)

        trainer = METHODS[args.method](**vars(args))
        trainer.run()

if __name__ == "__main__":
    main()
