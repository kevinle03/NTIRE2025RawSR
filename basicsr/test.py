import logging
import torch
import os
import math
from os import path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0,parent_dir)
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs,tensor2img
from basicsr.utils.options import dict2str, parse_options
from dataloader_moa import *
from torch.nn import functional as F

def pre_process(img, scale, pre_pad):
    """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
    """
    # pre_pad
    if pre_pad != 0:
        img = F.pad(img, (0, pre_pad, 0, pre_pad), 'reflect')
    # mod pad for divisible borders
    mod_pad_h, mod_pad_w = 0, 0
    _, h, w = img.size()
    if (h % scale != 0):
        mod_pad_h = (scale - h % scale)
    if (w % scale != 0):
        mod_pad_w = (scale - w % scale)
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return img, mod_pad_h, mod_pad_w

def tile_process(img, model, scale, tile_size, tile_pad, raw_img_max):
    """
    Processes an input image (C, H, W) by dividing it into tiles, upscaling each tile using a model,
    and then merging the processed tiles back into a single output image.

    Parameters:
    - img (torch.Tensor): Input image tensor of shape (C, H, W), values in range [0,1].
    - model (torch.nn.Module): The deep learning model used for upscaling.
    - scale (int): The upscaling factor.
    - tile_size (int): The size of each tile.
    - tile_pad (int): The padding applied to tiles.

    Returns:
    - torch.Tensor: The processed and upscaled output image in (C, H, W), range **[0,255]**.
    """
    _, height, width = img.shape
    output_height = height * scale
    output_width = width * scale

    # Initialize output image (keep float32, convert to 255 at the end)
    num_channels = 4
    # num_channels = 3
    output = torch.zeros((num_channels, output_height, output_width), dtype=torch.float32, device=img.device)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # Loop through all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile offsets
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            # Define the input tile area within the original image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # Define the padded tile area
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # Extract the input tile with padding
            input_tile = img[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # Model expects (B, C, H, W) format
            input_tile = input_tile.unsqueeze(0)  # Add batch dimension

            # Perform upscaling on the tile
            try:
                with torch.no_grad():
                    model.feed_test_data(input_tile)
                    model.test()

                    visuals = model.get_current_visuals()
                    output_tile = visuals['result']  # Output remains (B, C, H, W)
                    
                    # Remove batch dimension
                    output_tile = output_tile.squeeze(0)

                    # Release memory
                    del model.lq
                    del model.output
                    torch.cuda.empty_cache()

            except RuntimeError as error:
                print(f'Error processing tile {y * tiles_x + x + 1}/{tiles_x * tiles_y}: {error}')
                continue  # Skip this tile if an error occurs
            
            print(f'Processing Tile {y * tiles_x + x + 1}/{tiles_x * tiles_y}')

            # Define the output tile position in the final image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Define the valid (unpadded) portion of the output tile
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * scale

            # Place the processed tile into the output image
            output[:, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    # Convert from [0,1] to [0,max_val] before returning
    output = (output * raw_img_max).clamp(0, raw_img_max)
    # output = (output * 255).clamp(0, 255).byte()  # Ensure valid range and convert to uint8

    return output  # Returns tensor in (C, H, W), range [0,255]

def post_process(img, scale, pre_pad, mod_pad_h, mod_pad_w):

    # remove extra pad
    _, h, w = img.size()
    img = img[:, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
    # remove prepad
    if pre_pad != 0:
        _, h, w = img.size()
        img = img[:, 0:h - pre_pad * scale, 0:w - pre_pad * scale]
    return img

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    # test_loaders = []
    # for _, dataset_opt in sorted(opt['datasets'].items()):
    #     test_set = build_dataset(dataset_opt)
    #     test_loader = build_dataloader(
    #         test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
    #     logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    #     test_loaders.append(test_loader)
    
    # DIV2K dataset and loader:
    test_set = TestSetLoader_raw(args)
    dataloader = DataLoader(
                dataset=test_set,
                batch_size=opt['datasets']['train']['batch_size_per_gpu'] * torch.cuda.device_count(),
                shuffle=False,
                num_workers=4,
            )
    # create model
    # /scratch/ll5484/MambaIRV2/MambaIR-main/experiments/MambaIRv2_SR_x4_archived_20250323_121945/models
    model = build_model(opt)
    # if use_pbar:
    #     pbar = tqdm(total=len(dataloader), unit='image')
    output_dir = os.path.join(args.data_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for imgs, paths, raw_img_maxes in dataloader:
        for test_img, img_path, raw_img_max in zip(imgs, paths, raw_img_maxes):
            # sr_image = tile_process(test_img, model, opt['scale'], args.img_size, 10)
            pre_pad = 5
            pre_process_image, mod_pad_h, mod_path_w = pre_process(test_img, opt['scale'],pre_pad)
            processed_img = tile_process(pre_process_image, model, opt['scale'], args.img_size, args.img_size, raw_img_max)
            sr_image = post_process(processed_img, opt['scale'], pre_pad, mod_pad_h, mod_path_w)
            
            if len(sr_image.shape) == 4:  # (B, C, H, W) case
                sr_image = sr_image[0]
                
            sr_image = sr_image.cpu().detach().numpy().astype(np.uint16)
            sr_image = np.transpose(sr_image, (1, 2, 0))
            if not opt["raw"]:
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
            img_name = os.path.basename(img_path)
            # img_name = img_name[0:4]+img_name[6:]
            # Save the transformed image
            output_path = os.path.join(output_dir, img_name)
            if opt['raw']:
                np.savez(output_path, raw=sr_image, max_val=raw_img_max)
            else:
                cv2.imwrite(output_path, sr_image)
            del sr_image
            print(f"Image saved at {output_path}")
                
    # for test_loader in test_loaders:
    #     test_set_name = test_loader.dataset.opt['name']
    #     logger.info(f'Testing {test_set_name}...')
    #     model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
