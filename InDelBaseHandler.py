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