import torch
import numpy as np

from collections import namedtuple

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

VIPER_SEMANTIC_LABELS= [
 
  #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
 Label('unlabeled'	               , 0	,     255,  'void'            , 0       , False,          True,              (0,	   0,   0) ),	        	 
 Label('ambiguous'	               , 1	,     255,  'void'            , 0       , False,          True,             (111,  74,   0)    ),      	 
 Label('sky'	                   , 2	,       0,	'sky'             , 5       , False,          False,               ( 70, 130, 180) ),	             
 Label('road'	                   , 3	,       1,	'flat'            , 1       , False,          False,               (128,  64, 128) ),	             
 Label('sidewalk'	               , 4	,       2,	'flat'            , 1       , False,          False,               (244,  35, 232) ),	             
 Label('railtrack'	               , 5	,     255,  'flat'            , 1       , False,          True,             (230, 150, 140)    ),      	 
 Label('terrain'                   , 6	,       3,	'nature'          , 4       , False,          False,               (152, 251, 152) ),	             
 Label('tree'	                   , 7	,       4,	'nature'          , 4       , False,          False,               ( 87, 182,  35) ),	             
 Label('vegetation'	               , 8	,       5,	'nature'          , 4       , False,          False,               ( 35, 142,  35) ),	             
 Label('building'	               , 9	,       6,	'construction'    , 2       , False,          False,            ( 70,  70,  70)    ),              
  Label('infrastructure'           , 10,       7,	'construction'    , 2       , False,          False,               (153, 153, 153) ),	             
  Label('fence'	                   , 11,       8,	'construction'    , 2       , False,          False,               (190, 153, 153) ),	             
  Label('billboard'	               , 12,       9,	'construction'    , 2       , False,          False,             (150,  20,  20)   ),                
  Label('trafficlight'	           , 13,      10,   'object'          , 3       , True,           False,            (250, 170, 30)     ),      	     
  Label('trafficsign'	           , 14,      11,	'object'          , 3       , False,          False,            (220, 220,  0)     ),                
  Label('mobilebarrier'	           , 15,      12,   'object'          , 3       , False,          False,             (180, 180, 100)   ),       	     
  Label('firehydrant'	           , 16,      13,   'object'          , 3       , True,           False,             (173, 153, 153)   ),       	     
  Label('chair'	                   , 17,      14,   'object'          , 3       , True,           False,             (168, 153, 153)   ),       	     
  Label('trash'	                   , 18,      15,	'object'          , 3       , False,          False,            ( 81,   0,  21)    ),                
  Label('trashcan'	               , 19,      16,   'object'          , 3       , True,           False,             ( 81,   0,  81)   ),       	     
  Label('person'	               , 20,      17,   'human'           , 6       , True,           False,             (220,  20,  60)   ),       	     
  Label('animal'	               , 21,     255,   'nature'          , 4       , False,           True,             (255,   0,	0)     ),	        	 
  Label('bicycle'	               , 22,     255,   'vehicle'         , 7       , False,           True,           (119,  11,  32)     ),      	     
  Label('motorcycle'	           , 23,      18,   'vehicle'         , 7       , True,            False,            (  0,   0, 230)   ),       	     
  Label('car'	                   , 24,      19,   'vehicle'         , 7       , True,            False,            (  0,   0, 142)   ),       	     
  Label('van'	                   , 25,      20,   'vehicle'         , 7       , True,            False,            (  0,  80, 100)   ),       	     
  Label('bus'	                   , 26,      21,   'vehicle'         , 7       , True,            False,            (  0,  60, 100)   ),       	     
  Label('truck'	                   , 27,      22,   'vehicle'         , 7       , True,            False,            (  0,   0,  70)   ),       	     
  Label('trailer'	               , 28,     255,   'vehicle'         , 7       , False,             True,           (  0,   0,  90)   ),      	     
  Label('train'	                   , 29,     255,   'vehicle'         , 7       , False,             True,           (  0,  80, 100)   ),      	     
  Label('plane'	                   , 30,     255,   'vehicle'         , 7       , False,             True,           (  0, 100, 100)   ),      	     
  Label('boat'	                   , 31,     255,   'vehicle'         , 7       , False,             True,           ( 50,   0,  90)   ),      	     
]

CITYSCAPES_SEMANTIC_LABELS = [
    #       name                     id    trainId  category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , 34 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    #ORIGINAL: Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def getSemanticLabels(dataset_name):
    if "cityscapes" in dataset_name:
        return CITYSCAPES_SEMANTIC_LABELS
    elif ("viper" in dataset_name) or ("pfd" in dataset_name):
        return VIPER_SEMANTIC_LABELS
    else:
        return None

def getNumSemanticLabelIds(dataset_name, are_train_ids):
    N = None
    if "cityscapes" in dataset_name:
        if are_train_ids:
            N = 19
        else:
            N = 35
    elif ("viper" in dataset_name): # or ("pfd" in dataset_name):
        if are_train_ids:
            N = 23
        else:
            N = 32
    elif "pfd" in dataset_name:
        if are_train_ids:
            N = 19 #21
        else:
            N = 35 #25 There are 25 classes but we use 35, because the make compatibility with cityscapes easier
    elif "mseg" in dataset_name:
        if are_train_ids:
            N = 194
        else:
            N = 194
    return N


def labelColorMap(dataset_name, are_train_ids=True):
    labels = getSemanticLabels(dataset_name)

    cmap = []
    for label in labels:
        if label.ignoreInEval and are_train_ids:
            continue
        else:
            cmap.append(label.color)
    cmap = np.array(cmap, dtype=np.uint8)
    return cmap

def labelMapToColor(seg_map, dataset_name, are_train_ids=True):
    shape = list(seg_map.shape)
    assert (len(shape) == 4) or (len(shape) == 5)
    cmap = labelColorMap(dataset_name, are_train_ids=are_train_ids)
    
    shape[-3] = 3
    color_image = torch.ByteTensor(*shape).fill_(0).cuda()

    for label in range(0, cmap.shape[0]):
        mask = seg_map == label 
        if len(shape) == 4:
            color_image[:,0:1,:,:].masked_fill_(mask, cmap[label][0])
            color_image[:,1:2,:,:].masked_fill_(mask, cmap[label][1])
            color_image[:,2:3,:,:].masked_fill_(mask, cmap[label][2])
        elif len(shape) == 5:
            color_image[:,:,0:1,:,:].masked_fill_(mask, cmap[label][0])
            color_image[:,:,1:2,:,:].masked_fill_(mask, cmap[label][1])
            color_image[:,:,2:3,:,:].masked_fill_(mask, cmap[label][2])
    return color_image

def labelMapToOneHot(label_map, dataset_name, to_train_ids=True, inst_map=None):
    contain_dontcare_label = False
    if to_train_ids:
        contain_dontcare_label = True

    N = getNumSemanticLabelIds(dataset_name, to_train_ids)

    # create one-hot label map 
    shape = list(label_map.shape)
    assert (len(shape) == 4) or (len(shape) == 5)

    scatter_dim = 1
    if len(shape) == 5:
        scatter_dim = 2

    nc = N + 1 if contain_dontcare_label else N
    shape[scatter_dim] = nc

    input_label = torch.FloatTensor(*shape).zero_().cuda()
    input_semantics = input_label.scatter_(scatter_dim, label_map.long(), 1.0)

    # concatenate instance map if it exists
    if inst_map is not None:
        instance_edge_map = get_edges(inst_map, scatter_dim)
        instance_edge_map, _ = torch.max(instance_edge_map, dim=scatter_dim, keepdim=True)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=scatter_dim)

    return input_semantics

def get_edges(t, scatter_dim):
    edge = torch.ByteTensor(t.size()).zero_().cuda()
    if scatter_dim == 1:
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()
    elif scatter_dim == 2:
        edge[:,:, :, :, 1:] = edge[:,:, :, :, 1:] | (t[:,:, :, :, 1:] != t[:,:, :, :, :-1])
        edge[:,:, :, :, :-1] = edge[:,:, :, :, :-1] | (t[:,:, :, :, 1:] != t[:,:, :, :, :-1])
        edge[:,:, :, 1:, :] = edge[:,:, :, 1:, :] | (t[:,:, :, 1:, :] != t[:,:, :, :-1, :])
        edge[:,:, :, :-1, :] = edge[:,:, :, :-1, :] | (t[:,:, :, 1:, :] != t[:,:, :, :-1, :])
        return edge.float()
