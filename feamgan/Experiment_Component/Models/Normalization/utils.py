import torch.nn as nn

from apex import parallel

# Returns a function that creates a standard normalization function
def getNormLayer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = nn.utils.spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if norm_type == '' or subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'syncbatch':
            norm_layer = parallel.SyncBatchNorm(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'group':
            norm_layer = nn.GroupNorm(8, get_out_channel(layer), eps=1e-05, affine=False)
        elif subnorm_type == 'groupA':
            norm_layer = nn.GroupNorm(8, get_out_channel(layer), eps=1e-05, affine=True)
        elif subnorm_type == 'none':
            norm_layer = nn.Identity()
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

