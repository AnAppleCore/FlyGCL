import time
from configuration import config
from methods.default import Trainer
from methods.buffer import BufferTrainer

# for imagenet-r
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

methods = {'Default': Trainer, 'Buffer': BufferTrainer}

def main():

    args = config.base_parser()
    seed_lst = args.seeds
    print('>>>>>>>running for seed: {}'.format(seed_lst))
    for seed in seed_lst:
        setattr(args, 'rnd_seed', seed)
        print(args)

        trainer = methods[args.mode](**vars(args))
        trainer.run()

if __name__ == "__main__":
    main()
    time.sleep(3)