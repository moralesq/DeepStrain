from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        parser.add_argument('--isTrain', type=bool, default=False, help='load for training')
        parser.add_argument('--pretrained_models_netS', type=str, default='./pretrained_models/carson_Jan2021.h5', help=' ')
        parser.add_argument('--pretrained_models_netME', type=str, default='./pretrained_models/carmenJan2021.h5', help=' ')
  
        return parser