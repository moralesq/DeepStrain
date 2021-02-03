import os
import argparse
import tensorflow

class BaseOptions():
    
    def __init__(self):
        self.initialized = False
                
    def initialize(self, parser):
        """This class defines options used during both training and test time.
        """
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--dataformat', required=True, help='path to images')
        parser.add_argument('--max_dataset_size', type=float, default=float("inf"), help=' ')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
       
        # data parameters
        parser.add_argument('--preprocess', type=str, default='reshape_to_carson_crop_zscore', help=' ')
        parser.add_argument('--image_shape', type=tuple, default=(128,128,1), help=' ')
        parser.add_argument('--volume_shape', type=tuple, default=(128,128,16,1), help=' ')
        parser.add_argument('--nlabels', type=int, default=4, help='number of tissue classes')
        
        # data parameters
        parser.add_argument('--order', type=int, default=1, help='resampling order')
        parser.add_argument('--mode', type=str, default='nearest', help='resampling mode')
        parser.add_argument('--in_plane_resolution_mm', type=float, default=1.25, help='resample to in_plane_resolution_mm')
        parser.add_argument('--number_of_slices', type=int, default=None, help='resample to number_of_slices')
        parser.add_argument('--slice_thickness_mm', type=float, default=None, help='resample to slice_thickness_mm')

        # special tasks
        self.initialized = True
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # get the basic options
        opt, _ = parser.parse_known_args()
        
        # save and return the parser
        self.parser = parser
        return parser.parse_args()
            
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.opt = opt
        return self.opt
