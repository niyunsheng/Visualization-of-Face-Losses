import argparse
import glob
import imageio
import os
import shutil


def create_gif(image_list, gif_name, duration=1):
    frames = []
    for image_name in image_list:
        if image_name.endswith('.png'):
            frames.append(imageio.imread(image_name))
    last_image = image_list[-1]
    shutil.copy(last_image, os.path.join('images',os.path.basename(last_image)))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)


def main():
    parser = argparse.ArgumentParser("PyTorch faceloss sample")
    arg = parser.add_argument
    arg('--m', type=str, default=0.01)
    arg('--start-epoch', type=int, default=0)
    arg('--end-epoch', type=int, default=20)
    arg('--data-dir', type=str, default="weights/")
    arg('--lossname', type=str, default="softmax")
    arg('--save-dir', type=str, default="images")
    arg('--duration', type=float, default=1)
    args = parser.parse_args()

    image_list=[ os.path.join(args.data_dir, args.lossname, 
                                '{}_train_epoch{:02}.png'.format(args.lossname, epoch_id)) 
                                for epoch_id in range(args.start_epoch,args.end_epoch)]
    create_gif(image_list, os.path.join(args.save_dir,'{}_train.gif'.format(args.lossname)))

    image_list=[ os.path.join(args.data_dir,args.lossname,
                                '{}_test_epoch{:02}.png'.format(args.lossname, epoch_id)) 
                                for epoch_id in range(args.start_epoch,args.end_epoch)]
    create_gif(image_list, os.path.join(args.save_dir,'{}_test.gif'.format(args.lossname)))
 
if __name__ == "__main__":
    main()