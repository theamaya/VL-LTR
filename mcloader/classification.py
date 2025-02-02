import os
import os.path as osp
import pickle

from torch.utils.data import Dataset

from .imagenet import ImageNet
from .image_list import ImageList
from .inat import iNat
from .preprocess import SentPreProcessor


def get_sentence_tokens(dataset: str, desc_path, context_length, include_wiki, prompts):
    print('using clip text tokens splitted by sentence')
    if prompts=='coop':
        print('using coop prompts')
        if include_wiki:
            print('with wikiprompts')
        preprocessor = SentPreProcessor(root=desc_path, include_wiki=include_wiki, prompts=prompts, dataset=dataset) #self, root, include_wiki= True, prompts= 'all', dataset: str="INAT"
        texts = preprocessor.get_clip_text()
        texts = preprocessor.split_sent(texts)
        text_tokens = preprocessor.tokenize(texts, context_length=context_length)
        return text_tokens

    cache_root = 'cached'
    if include_wiki:
        print('wiki descriptions')
        cache_path = osp.join(cache_root, '%s_desc_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens
    elif prompts=='all':
        print('multiple prompts')
        cache_path = osp.join(cache_root, '%s_prompts_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_prompts_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens
    elif prompts=='single':
        print('single prompt')
        cache_path = osp.join(cache_root, '%s_single_prompt_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_single_prompt_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens
    elif prompts=='half':
        print('40 prompts')
        cache_path = osp.join(cache_root, '%s_40_prompt_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_40_prompt_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens
    elif prompts=='coop':
        print('coop prompts')
        cache_path = osp.join(cache_root, '%s_coop_prompt_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_coop_prompt_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens
    elif prompts=='cupl':
        print('cupl prompts')
        cache_path = osp.join(cache_root, '%s_cupl_prompt_text_sent.pkl' % dataset)
        clip_token_path = osp.join(cache_root, '%s_cupl_prompt_text_tokens.pkl' % dataset)
        if osp.exists(clip_token_path):
            with open(clip_token_path, 'rb') as f:
                text_tokens = pickle.load(f)
            return text_tokens

    preprocessor = SentPreProcessor(root=desc_path, include_wiki=include_wiki, prompts=prompts, dataset=dataset) #self, root, include_wiki= True, prompts= 'all', dataset: str="INAT"
    if not osp.exists(cache_path):
        os.makedirs(cache_root, exist_ok=True)
        texts = preprocessor.get_clip_text()
        texts = preprocessor.split_sent(texts)
        with open(cache_path, 'wb') as f:
            pickle.dump(texts, f)
    else:
        with open(cache_path, 'rb') as f:
            texts = pickle.load(f)

    text_tokens = preprocessor.tokenize(texts, context_length=context_length)
    with open(clip_token_path, 'wb') as f:
        pickle.dump(text_tokens, f)
    return text_tokens


class ClassificationDataset(Dataset):
    """Dataset for classification.
    """

    def __init__(self, dataset='IMNET', split='train', nb_classes=1000,
                 desc_path='', 
                 include_wiki= True, 
                 prompts= 'all',
                 context_length=0, pipeline=None, select=False):
        assert dataset in ['PLACES_LT', "IMNET", "IMNET_LT", "INAT"]
        self.nb_classes = nb_classes
        if dataset == 'IMNET':
            self.data_source = ImageNet(root='data/imagenet/%s' % split,
                                        list_file='data/imagenet/meta/%s.txt' % split,
                                        select=select)
        elif dataset == 'IMNET_LT':
            self.data_source = ImageNet(root='data/imagenet',
                                        list_file='data/imagenet/ImageNet_LT_%s.txt' % split)
        elif dataset == 'INAT':
            if split=='test':
                self.data_source = iNat(root='data/iNat/',
                                    json_file='val2018.json',
                                    select=select)
            else:
                self.data_source = iNat(root='data/iNat/',
                                    json_file='%s2018.json' % split,
                                    select=select)
            # self.data_source = ImageList(root='/l/users/amaya.dharmasiri/data/iNat/',
            #                             list_file='/l/users/amaya.dharmasiri/data/iNat/inat2018_%s.txt' % split,
            #                             select=select)
        elif dataset == 'PLACES_LT':
            self.data_source = ImageList(root='data/places',
                                        list_file='data/places/Places_LT_%s.txt' % split,
                                        select=select)
        
        self.text_tokens = get_sentence_tokens(dataset, desc_path, context_length, include_wiki, prompts)
        self.end_idxs = [len(sents) for sents in self.text_tokens]

        self.pipeline = pipeline
        assert self.data_source.labels is not None
        self.targets = self.data_source.labels

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)

        return img, target
