import pandas as pd
import numpy as np
import sys
from InDelBaseHandler import PlusPlusHandler, MinusPlusHandler, PlusMinusHandler, MinusMinusHandler


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