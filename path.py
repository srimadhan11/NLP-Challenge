import os

from helper import mkdir_p


class Path:
    '''Manage paths of all the required files'''
    def __init__(self, nlp_dir, phase=1, out_ver=0):
        nlp_dir = nlp_dir
        out_dir = os.path.join(nlp_dir, f'output{out_ver}')

        '''
        Brief discription of paths being used:
            train.csv   - training set
            phase_dir   - directory, where phase related output is generated
            test.csv    - test set, for the respective phase
            answer_dir  - directory, where the answers associated with the phase is generated
            answer.txt  - prediction file for given test set

            model       - model's final learned weights
            model_dir   - model's weight checkpoint directory
            model.bkp   - model's weight checkpoint files

            hi_lang     - Hindi Lang object
            en_lang     - English Lang object
            hi_tkns     - Hindi parsed training tokens
            en_tkns     - English parsed training tokens

            dev_hi_tkns - Hindi parsed development set tokens
            dev_en_tkns - English parsed development set tokens

            split_dev   - Development set after splitting train.csv
            split_train - Training set after splitting train.csv
        '''
        
        self.paths = {
            'train.csv' : f'{nlp_dir}/train.csv',
            'phase_dir' : f'{nlp_dir}/phase{phase}',
            'test.csv'  : f'{nlp_dir}/phase{phase}/test.csv',
            'answer_dir': f'{nlp_dir}/phase{phase}/answer',
            'answer.txt': f'{nlp_dir}/phase{phase}/answer/{{}}.txt',

            'model'    : f'{out_dir}/model.pth',
            'model_dir': f'{out_dir}/model',
            'model.bkp': f'{out_dir}/model/{{}}.pth',

            'hi_lang': f'{out_dir}/hi_lang.pkl',
            'en_lang': f'{out_dir}/en_lang.pkl',
            'hi_tkns': f'{out_dir}/hi_tkns.npy',
            'en_tkns': f'{out_dir}/en_tkns.npy',

            'dev_hi_tkns': f'{out_dir}/dev_hi_tkns.npy',
            'dev_en_tkns': f'{out_dir}/dev_en_tkns.npy',

            'split_dev'  : f'{out_dir}/split_dev.pkl',
            'split_train': f'{out_dir}/split_train.pkl'
        }
        mkdir_p(out_dir)
        mkdir_p(self.paths['phase_dir'])
        mkdir_p(self.paths['answer_dir'])
        mkdir_p(self.paths['model_dir'])
        pass

    def __call__(self, filename, *param):
        filepath = self.paths[filename]
        if '{}' in filepath:
            assert param is not None
            return filepath.format(*param)
        else:
            return filepath
        pass
