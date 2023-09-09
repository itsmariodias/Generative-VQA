import os
import pickle
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from data import ImageDetectionsField, TextField, COCOM2Tranformer
from vqa.data.build import make_dataloader
from vqa.modules import *


@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    m2_transformer_info_list = []
    ############## Start of Preparation for text_field ##############
    # Gavin: copied from M2Transformer, vocab is used for preparing output for VLBERT vqa generation
    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path="../M2TransformerData/coco_detections.hdf5", max_detections=50,
                                       load_in_tmp=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False, fix_length=10)
    # The image section is not needed, but it requires less modification of the code
    dataset = COCOM2Tranformer(image_field, text_field, 'coco/images/', "../M2TransformerData/data/annotations",
                               "../M2TransformerData/data/annotations")
    train_dataset, val_dataset, test_dataset = dataset.splits
    if not os.path.isfile('vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    ############### End of Preparation for text_field ###############
    m2_transformer_info_list.append(text_field)

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False, m2_transformer_info_list=m2_transformer_info_list)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    q_ids = []
    answer_ids = []
    model.eval()
    cur_id = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        q_ids.extend([test_database[id]['question_id'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        batch = to_cuda(batch)
        output = model(*batch)
        answer_ids.extend(output['decoder_output'].argmax(dim=1).detach().cpu().tolist())
        cur_id += bs

    result = [{'question_id': q_id, 'answer': test_dataset.answer_vocab[a_id]} for q_id, a_id in zip(q_ids, answer_ids)]

    cfg_name = os.path.splitext(os.path.basename(args.cfg))[0]
    result_json_path = os.path.join(save_path, '{}_vqa2_{}.json'.format(cfg_name if save_name is None else save_name,
                                                                        config.DATASET.TEST_IMAGE_SET))
    with open(result_json_path, 'w') as f:
        json.dump(result, f)
    print('result json saved to {}.'.format(result_json_path))
    return result_json_path
