import dill
from sklearn.base import BaseEstimator
import xgboost as xgb
import pandas as pd
import numpy as np
import sys
from abc import ABCMeta, abstractmethod


class InDelBaseHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_base(self, ref_base, change_base_1, change_base_2):
        pass

    @staticmethod
    def get_operator(top_base):
        operator_str = top_base[0]
        change_base = top_base[1:]
        return operator_str, change_base

    @staticmethod
    def get_ref_base(change_base, org_ref_base):
        if change_base in org_ref_base:
            ref_base = org_ref_base
        else:
            ref_base = org_ref_base[0] + change_base[1:]
        return ref_base

    @staticmethod
    def plus_base(ref_base, change_base):
        mut_base = ref_base[0] + change_base[1:] + ref_base[1:]
        return mut_base

    @staticmethod
    def minus_base(ref_base, change_base):
        mut_base = ref_base.replace(change_base[1:], '', 1)
        return mut_base


class PlusPlusHandler(InDelBaseHandler):

    def get_base(self, ref_base, change_base_1, change_base_2):
        mutant_base_1 = self.plus_base(ref_base, change_base_1)
        mutant_base_2 = self.plus_base(ref_base, change_base_2)
        return ref_base, mutant_base_1, mutant_base_2


class MinusPlusHandler(InDelBaseHandler):
    def get_base(self, ref_base, change_base_1, change_base_2):
        ref_base = self.get_ref_base(change_base_1, ref_base)
        mutant_base_1 = self.minus_base(ref_base, change_base_1)
        mutant_base_2 = self.plus_base(ref_base, change_base_2)
        return ref_base, mutant_base_1, mutant_base_2


class PlusMinusHandler(InDelBaseHandler):
    def get_base(self, ref_base, change_base_1, change_base_2):
        ref_base = self.get_ref_base(change_base_2, ref_base)
        mutant_base_1 = self.plus_base(ref_base, change_base_1)
        mutant_base_2 = self.minus_base(ref_base, change_base_2)
        return ref_base, mutant_base_1, mutant_base_2


class MinusMinusHandler(InDelBaseHandler):
    def get_base(self, ref_base, change_base_1, change_base_2):
        ref_base = self.get_ref_base(change_base_1, ref_base)
        ref_base = self.get_ref_base(change_base_2, ref_base)
        mutant_base_1 = self.minus_base(ref_base, change_base_1)
        mutant_base_2 = self.minus_base(ref_base, change_base_2)
        return ref_base, mutant_base_1, mutant_base_2

def pup_reader(txt_path, label=None, labeled=True):
    if labeled:
        label = [item for item in label.split('\n')]
        while '' in label:
            label.remove('')
    concatenated = pd.read_csv(txt_path, header=None, dtype=float, delimiter='\t').values
    concatenated = np.array(concatenated).astype(np.float)
    nans = np.array((np.isnan(concatenated) | np.isinf(concatenated)).nonzero()).T.astype(np.int)
    if len(nans) > 0:
        print('NaNs detected. PLEASE FIX YOUR INPUT!!!')

        for a, b in nans:
            print('Number on row %6d feature %4d is %s.' % (a, b, str(concatenated[a, b])))

        print('Program will continue, but the model will be risky.')
        concatenated = np.nan_to_num(concatenated)
    if labeled:
        return np.array(label).astype(np.int), concatenated
    else:
        return concatenated


class ModelPostProcess(object):

    def __init__(self):
        self.true_points = None
        self.idf_points = None
        self.skip_rows = 0
        self.somatic_result = None
        self.true_label = None
        self.idf_result = None

    def get_file_header_line(self, file_path):
        self.skip_rows = 0
        with open(file_path) as File:
            for line in File:
                if line[0] in '#':
                    self.skip_rows += 1
                else:
                    break
        return self.skip_rows

    def get_header(self, feature_path, file_path):
        skip_rows = self.get_file_header_line(file_path=file_path)
        pup_cols = pd.read_table(
            file_path, nrows=0, header=None, skiprows=skip_rows).shape[1]
        m_features = []
        s_features = []
        s2_features = []
        with open(feature_path) as F:
            for line in F:
                platform_type = line.strip('\n').split('\t')
                if platform_type[3] == 'M':
                    m_features.append(line.split('\t')[1])
                if platform_type[3] == 'S':
                    s_features.append(line.split('\t')[1])
                if platform_type[3] == 'S2':
                    s2_features.append(line.split('\t')[1])
        m_cols = len(m_features)
        s_cols = len(s_features)
        n_platform = (pup_cols - m_cols) / s_cols
        if n_platform == 1:
            return m_features + s_features
        elif n_platform == 2:
            n_features = map(lambda x: 'N_' + x, s_features)
            t_features = map(lambda x: 'T_' + x, s_features)
            print ("s2_features", s2_features)
            return m_features + n_features + t_features + s2_features

    def get_true_points(self, label, feature_path, file_path):
        skips = [i for i, x in enumerate(label) if x == 0]
        col_name = self.get_header(feature_path, file_path)
        self.true_points = pd.read_table(
            file_path,
            header=None,
            skiprows=skips,
            names=col_name)
        return self.true_points

    def get_true_label(self, label):
        self.true_label = [label[i] for i, x in enumerate(label) if x != 0]
        return self.true_label

    def get_idf_points_with_prob(self, label, prob, df):
        sel_prob = [prob[i] for i, x in enumerate(label) if x != 0]
        sel_prob_s = pd.Series(sel_prob, name='prob')
        self.idf_points = pd.concat([df, sel_prob_s], axis=1)
        return self.idf_points

    def get_somatic_result(self, true_points, keep_path):
        df_keep = pd.read_table(keep_path, header=None, skiprows=0)
        df_chr = map(lambda x, y: str(x) + "_" + str(y),
                     true_points.values[:, 0],
                     true_points.values[:, 1])
        df_keep_chr = map(lambda x, y: str(x) + "_" + str(y),
                          df_keep.values[:, 0], df_keep.values[:, 1])
        chr_pos = np.concatenate([df_chr, df_keep_chr])
        u, indices = np.unique(chr_pos, return_index=True)
        all_df = pd.concat([true_points, df_keep], ignore_index=True)
        self.somatic_result = pd.DataFrame(data=[all_df.loc[index].values
                                                 for index in all_df.index if index in indices])
        return self.somatic_result

    @staticmethod
    def get_operator(top_base):
        operator_str = top_base[0]
        change_base = top_base[1:]
        return operator_str, change_base

    def get_idf_result(self, true_label, idf_points):
        gt_list = []
        for i in idf_points.index:
            if true_label[i] == 3:
                top_base_1 = idf_points.iloc[i]['top1_base']
                top_base_2 = idf_points.iloc[i]['top2_base']
                collections_handler = {'++': PlusPlusHandler,
                                       '-+': MinusPlusHandler,
                                       '+-': PlusMinusHandler,
                                       '--': MinusMinusHandler}
                ref_base = idf_points.iloc[i]['ref_base']
                if top_base_1 != '0' and top_base_2 != '0':
                    operator_1, change_base_1 = self.get_operator(top_base_1)
                    operator_2, change_base_2 = self.get_operator(top_base_2)
                    if operator_1 == '+' and operator_2 == '+':
                        handler = collections_handler['++']
                        ref_base, mutant_base_1, mutant_base_2 = handler(). \
                            get_base(ref_base, change_base_1, change_base_2)

                    elif operator_1 == '+' and operator_2 == '-':
                        handler = collections_handler['+-']
                        ref_base, mutant_base_1, mutant_base_2 = handler(). \
                            get_base(ref_base, change_base_1, change_base_2)

                    elif operator_1 == '-' and operator_2 == '+':
                        handler = collections_handler['-+']
                        ref_base, mutant_base_1, mutant_base_2 = handler(). \
                            get_base(ref_base, change_base_1, change_base_2)

                    elif operator_1 == '-' and operator_2 == '-':
                        handler = collections_handler['--']
                        ref_base, mutant_base_1, mutant_base_2 = handler(). \
                            get_base(ref_base, change_base_1, change_base_2)
                    else:
                        print ('top base error ', top_base_1, top_base_2)
                        sys.exit(1)

                    # change ref_base
                    idf_points['ref_base'][i] = ref_base
                    # change mut_base
                    idf_points['mut_base'][i] = mutant_base_1 + ',' + mutant_base_2
                    # insert Gtype
                    gt_list.append('1|2')

                elif top_base_1 == '0' or top_base_2 == '0':
                    # insert Gtype
                    gt_list.append('0|1')
                else:
                    print ('top base error ', top_base_1, top_base_2)
                    sys.exit(1)

            elif true_label[i] == 2:
                # insert Gtype: heterozygous
                gt_list.append('0|1')

            elif true_label[i] == 1:
                # insert Gtype: homozygous
                gt_list.append('1|1')

            else:
                print ("GenoType code error: "+str(true_label[i]))
                sys.exit(1)
        gt_s = pd.Series(gt_list, name='GT')
        self.idf_result = pd.concat([idf_points, gt_s], axis=1)
        return self.idf_result


class dummy_km:

    '''
    a dummy k-means like interface which always return 0
    '''

    def __init__(self, n_comp, n_jobs):
        pass

    def fit(self, x):
        return self

    def predict(self, x):
        return np.zeros(x.shape[0])

    def fit_predict(self, x):
        return self.predict(x)


class GenomeVariantCaller(BaseEstimator):
    def __init__(self, config=None):
        self.classifiers = {}
        self.pup_reader = None
        self.name_dic = {}
        if config is None:
            param = {}
            param['booster'] = 'gbtree'
            param['silent'] = 1
            param['nthread'] = 36
            param['eval_metric'] = 'mlogloss'
            param['max_depth'] = 5
            param['min_child_weight'] = 1
            param['cluster_num'] = 1
            self.params = param
        else:
            self.params = config
        print (self.params)

    def predict_file(self, path, sel_path, feature_file, output_file, km=True):
        x = pup_reader(path, labeled=False)
        pred = self.predict(x, km=km)
        label = np.argmax(pred, axis=1)
        prob = np.max(pred, axis=1)
        post_process = ModelPostProcess()
        true_points = post_process.get_true_points(label, feature_file, sel_path)
        true_label = post_process.get_true_label(label)
        idf_result = post_process.get_idf_result(true_label, true_points)
        result_df = post_process.get_idf_points_with_prob(label, prob, idf_result)
        result_df.to_csv(output_file, sep='\t', header=False, index=False)
        return result_df

    def predict(self, x, norm=True, km=True):
        if norm:
            x -= self.matmin
            x /= self.matmax

        classified = self.km.predict(x)
        label = np.zeros(classified.shape + (self.params['num_class'],))

        for idx in set(classified):
            if hasattr(self.classifiers[idx], 'predict'):
                if km:
                    xg_pred = xgb.DMatrix(x[classified == idx])
                    pred = self.classifiers[idx].predict(xg_pred)
                else:
                    xg_pred = x[classified == idx]
                    pred = self.classifiers[idx].predict_proba(xg_pred)
                label[classified == idx] = pred
            else:
                label[classified == idx] = 0.0
                label[classified == idx, self.classifiers[idx]] = 1.0

        return label

    def load(self, path):
        with open(path, "rb") as f:
            self.classifiers, self.km, self.matmin, self.matmax, self.pup_reader, self.params = dill.load(f)
            if 'dummy_km' in str(self.km):
                self.km = dummy_km(1,1)
        return self


if __name__ == '__main__':
    pup_dir = '/mnt/d/python/learn/indel_GT/test/germline.idf.out.txt'
    sel_dir = '/mnt/d/python/learn/indel_GT/test/germline.idf.out.sel'
    model_dir = '/mnt/d/python/learn/indel_GT/test/germline.model'
    feature_dir = '/mnt/d/python/learn/indel_GT/test/idf-feature'
    out_dir = '/mnt/d/python/learn/indel_GT/test/out'
    # pup_dir = '/disk/haixiao/python-learn/gvc_pre_update/output/germline.idf.out.txt'
    # sel_dir = '/disk/haixiao/python-learn/gvc_pre_update/output/germline.idf.out.sel'
    # model_dir = '/disk/haixiao/python-learn/gvc_pre_6_13/Model/Depth/Hiseq/WGS/INDEL/Germline/60.model'
    # feature_dir = '/disk/haixiao/python-learn/gvc_pre_update/source//feature-txt/format-test/idf-feature'
    # out_dir = '/disk/haixiao/python-learn/gvc_pre_update/output/out'
    gvc = GenomeVariantCaller()
    model = gvc.load(model_dir)
    gvc.predict_file(pup_dir, sel_dir, feature_dir, out_dir)




