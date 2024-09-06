from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger


from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged', 'SUS']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'SUS':
        from .datasets.SUS import SUS
        return SUS(cfg, mode='train'), SUS(cfg, mode='val'), SUS(cfg, mode='test')

def get_model(cfg):

    ############# model_others ################
#  RGB_T
    if cfg['model_name'] == 'MFNet':
        from model_others.RGB_T.MFNet import MFNet
        return MFNet(6)

    if cfg['model_name'] == 'MMSMCNet':
        from model_others.RGB_T.MMSMCNet import nation
        return nation()

    if cfg['model_name'] == 'SGFNet':
        from model_others.RGB_T.SGFNet.SGFNet import SGFNet
        return SGFNet(6)

    if cfg['model_name'] == 'CMX':
        from model_others.RGB_T.CMX.models.builder import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'TSFANet-T':
        from model_others.RGB_T.MAINet import MAINet
        return MAINet(False)

    if cfg['model_name'] == 'GMNet':
        from model_others.RGB_T.GMNet import GMNet
        return GMNet(6)

    if cfg['model_name'] == 'CAINet':
        from model_others.RGB_T.CAINet import mobilenetGloRe3_CRRM_dule_arm_bou_att
        return mobilenetGloRe3_CRRM_dule_arm_bou_att(9)

    if cfg['model_name'] == 'EGFNet':
        from model_others.RGB_T.EGFNet import EGFNet
        return EGFNet(6)

    if cfg['model_name'] == 'RTFNet':
        from model_others.RGB_T.RTFNet import RTFNet
        return RTFNet(6)

    if cfg['model_name'] == 'SFAFMA':
        from model_others.RGB_T.SFAFMA import SFAFMA
        return SFAFMA(6)

    if cfg['model_name'] == 'EAEFNet':
        from model_others.RGB_T.EAEFNet import EAEFNet
        return EAEFNet(6)

    if cfg['model_name'] == 'CENet':
        from model_others.RGB_T.TSmodel import Teacher_model
        return Teacher_model(6)

    if cfg['model_name'] == 'MSIRNet':
        from model_others.RGB_T.MS_IRTNet_main.Convnextv2.builder import Convnextv2
        return Convnextv2()


# RGB
    if cfg['model_name'] == 'FCN_8s':
        from model_others.RGB.FCN import FCN
        return FCN(9)

    if cfg['model_name'] == 'PSPNet':
        from model_others.RGB.PSPNet import PSPNet
        return PSPNet(6)

    if cfg['model_name'] == 'PSANet':
        from model_others.RGB.PSANet import PSANet
        return PSANet(9)

    if cfg['model_name'] == 'DeeplabV3+':
        from model_others.RGB.DeeplabV3.modeling import deeplabv3plus_resnet101
        return deeplabv3plus_resnet101(9)


## student
    if cfg['model_name'] == 'CLNet_S_irseg':
        from model_others.RGB_T.CLNet_S import Model
        return Model()

    if cfg['model_name'] == 'CLNet_S_pst900':
        from proposed.CLNet.student.CLNet_S_pst900 import Model
        return Model()

    if cfg['model_name'] == 'TSFANet_S_irseg':
        from proposed.TSFANet.TSFANet_S_irseg import Student
        return Student()

    if cfg['model_name'] == 'TSFANet_S_pst900':
        from proposed.TSFANet.TSFANet_S_pst900 import Student
        return Student()


## teacher

    if cfg['model_name'] == 'CLNet_T_irseg':
        from proposed.CLNet.teacher.CLNet_T_irseg import Teacher
        return Teacher()

    if cfg['model_name'] == 'CLNet_T_pst900':
        from proposed.CLNet.teacher.CLNet_T_PST900 import Teacher
        return Teacher()

    if cfg['model_name'] == 'TSFANet_T_irseg':
        from proposed.TSFANet.TSFANet_T_irseg import Teacher
        return Teacher()

    if cfg['model_name'] == 'TSFANet_T_SUS':
        from proposed.TSFANet.TSFANet_T_SUS import Teacher
        return Teacher()

    if cfg['model_name'] == 'CLNet_T_SUS':
        from proposed.CLNet.teacher.CLNet_T_SUS import Teacher
        return Teacher()





#### KD
    if cfg['model_name'] == 'pst900_KD':
        from proposed.CLNet.distillation.student.CLNet_KD_pst900 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'irseg_KD':
        from proposed.CLNet.distillation.student.CLNet_KD_irseg import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'TSFANet_KD_irseg':
        from proposed.TSFANet.TSFANet_KD_irseg import Student
        return Student()

    if cfg['model_name'] == 'TSFANet_KD_pst900':
        from proposed.TSFANet.TSFANet_KD_pst900 import Student
        return Student()

    if cfg['model_name'] == 'TSFANet_KD_SUS':
        from proposed.TSFANet.TSFANet_KD_SUS import Student
        return Student()

    if cfg['model_name'] == 'MHFINet_KD_SUS':
        from proposed.CLNet.distillation.student.CLNet_KD_SUS import Student
        return Student(distillation=True)

    ## visualize
    if cfg['model_name'] == 'model5_b4_visualize':
        from proposed.CLNet.teacher import Teacher
        return Teacher()

## ablation1
    if cfg['model_name'] == 'withoutMHFI':
        from proposed.CLNet.teacher import Teacher
        return Teacher()

    if cfg['model_name'] == 'withoutDGSD':
        from proposed.CLNet.teacher.wo_DGSD import Teacher
        return Teacher()

    if cfg['model_name'] == 'base':
        from proposed.CLNet.teacher import Teacher
        return Teacher()

# TSFANet

    if cfg['model_name'] == 'baseline':
        from proposed.TSFANet.ablation.baseline import MAINet
        return MAINet()

    if cfg['model_name'] == 'TSFA':
        from proposed.TSFANet.ablation.TSFANet import MAINet
        return MAINet()

    if cfg['model_name'] == 'TSFA_pst900':
        from proposed.TSFANet.TSFANet_T_pst900 import MAINet
        return MAINet()


# proposed

    if cfg['model_name'] == 'model0':
        from proposed.teacher import Model
        return Model(name='base')

    if cfg['model_name'] == 'model1':
        from proposed.model1 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'model2':
        from proposed.model2 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'CMX2':
        from proposed.student import EncoderDecoder
        return EncoderDecoder()


# ConvNext_base

    if cfg['model_name'] == 'convnext_nano':
        from proposed.teacher import Model
        return Model(name='nano')

    if cfg['model_name'] == 'convnext_tiny':
        from proposed.teacher import Model
        return Model(name='tiny')


# ConvNext

    if cfg['model_name'] == 'Segformer_b0':
        from proposed.backbone_model.Segformer.build_model_b0 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'Segformer_b1':
        from proposed.backbone_model.Segformer.build_model_b1 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'Segformer_b2':
        from proposed.backbone_model.Segformer.build_model_b2 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'Segformer_b3':
        from proposed.backbone_model.Segformer.build_model_b3 import EncoderDecoder
        return EncoderDecoder()

    if cfg['model_name'] == 'Segformer_b4':
        from proposed.backbone_model.Segformer.build_model_b4 import EncoderDecoder
        return EncoderDecoder()

    # SUS



        
    # ablation
    if cfg['model_name'] == 'teacher':
        from proposed.teacher.teacher import Model
        return Model(name='base')

    if cfg['model_name'] == 'backbone_teacher':
        from proposed.teacher.back_bone import Model
        return Model(name='base')

    if cfg['model_name'] == 'student':
        from proposed.student.student import Model
        return Model(name='nano')

    # KD
    if cfg['model_name'] == 'KD_backbone':
        from proposed.KD.KD_backbone import Model
        return Model(name='nano')

    if cfg['model_name'] == 'KD_student':
        from proposed.KD.KD_mem import Model
        return Model(name='nano')

    # backbone
    if cfg['model_name'] == 'segformer_teacher':
        from proposed.backbone_model.Segformer.teacher import Model
        return Model()

    if cfg['model_name'] == 'segformer_student':
        from proposed.backbone_model.Segformer.student import Model
        return Model()

    if cfg['model_name'] == 'teacher2':
        from proposed.teacher2 import Model
        return Model(name='base')














