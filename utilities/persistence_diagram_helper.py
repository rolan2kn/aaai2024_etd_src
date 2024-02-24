
import numpy as np


class PersistenceDiagramHelper:
    @staticmethod
    def is_in_gudhi_format(PD):
        if PD is None:
            raise Exception("We expected a collection of persistence intervals, but none was retrieved")

        if type(PD) == dict:
            return False

        pd_size = len(PD)
        if pd_size < 1:
            raise Exception("We expected a non-empty collection of "
                            "persistence intervals, but an empty one was retrieved")
        first_interval = PD[0]

        if type(first_interval) in (tuple, list):
            items_size = len(first_interval)

            if items_size > 0 and len(first_interval[1]) == 2:
                return True

        return False

    @staticmethod
    def ripser_2_useful_pd(ripser_diags):
        if ripser_diags is None:
            raise Exception("We expected a collection of persistence intervals, but none was retrieved")

        tmp_diags = {}
        diags = []
        dgms_collections = ripser_diags
        if type(ripser_diags) == dict and "dgms" in ripser_diags:
            dgms_collections = ripser_diags["dgms"]

        for d, Hi in enumerate(dgms_collections):
            for pi in Hi:
                if d not in tmp_diags:
                    tmp_diags.update({d:[]})
                tmp_diags[d].append(pi)

        for d in sorted(tmp_diags):
            diags.append(tmp_diags[d])

        del dgms_collections
        del tmp_diags

        return diags

    @staticmethod
    def epd_from_pd_collection(iid_pd_collection):
        max_sizes = PersistenceDiagramHelper.get_maximum_sizes(iid_pd_collection)
        ED = []

        for dim in max_sizes:
            ED.append([[0, 0] for _ in range(max_sizes[dim])])

        for diags in iid_pd_collection:
            for dim, ddiags in enumerate(diags):
                for id, (b, d) in enumerate(ddiags):
                    ED[dim][id][0] += b
                    ED[dim][id][1] += d

        for dim, hgroup in enumerate(ED):
            for id, pi in enumerate(hgroup):
                ED[dim][id][0] /= max_sizes[dim]
                ED[dim][id][1] /= max_sizes[dim]

        return ED

    @staticmethod
    def from_curated_to_original(diag):
        odiag = []
        for d, ddiag in enumerate(diag):
            odiag.extend([(d, (pi[0],pi[1])) for pi in ddiag])

        return odiag

    @staticmethod
    def gudhi_2_useful_pd(gudhi_diags):
        if gudhi_diags is None:
            raise Exception("We expected a collection of persistence intervals, but none was retrieved")

        tmp_diags = {}
        diags = []
        for d, pi in gudhi_diags:
            if d not in tmp_diags:
                tmp_diags.update({d:[]})

            tmp_diags[d].append(pi)

        for d in sorted(tmp_diags):
            diags.append(tmp_diags[d])

        del tmp_diags
        return diags

    @staticmethod
    def get_maximal_dimension(full_data):
        if len(full_data) == 0:
            return None
        if hasattr(full_data, "keys"):
            keys = list(full_data.keys())
            '''
            By construction we know that all layers have the same number of homology groups
            If other data is provided you should homogenize all diagrams to have the same homology group cardinality
            '''
            first_layer_data = full_data[keys[0]]
            if 'dgms' not in first_layer_data:
                return None
            max_dim = len(first_layer_data["dgms"])
        else:
            max_size = PersistenceDiagramHelper.get_maximum_sizes(full_data)
            max_dim = len(max_size)

        return max_dim

    @staticmethod
    def truncate_diagram(PD, max_eps = None):
        '''
        Usually the functions to process persistence diagrams are not defined for immortal persistence
        intervals.
        However, immortal persistence intervals arise very often. Since immortal persistence intervals are
        related with
        very strong topological features, it could be a mistake to ignore them. For this reason, we truncate
        immortal
        persistence intervals to the maximum filtration value across all homology groups.

        :param PD: a collection of persistence diagrams, in each position i it holds the persistence
        diagram of
                        the i-th homology group.
        :param max_eps: the maximum filtration value. if it is None we compute across all homology groups
        :return: The diagrams without any immortal persistence interval
        '''
        PD = PersistenceDiagramHelper.homogenyze_diagram(PD)
        hgroups_number = len(PD)  # the number of homology groups
        hgroups = []

        if max_eps is None:
            max_eps = -1
            for d in range(hgroups_number):
                H_d = PD[d]  # we get the persistence intervals
                temp_hd = np.nan_to_num(H_d, posinf=0, nan=0)  # we detect the max value if we need
                max_eps = max(temp_hd.max(), max_eps)
                del temp_hd

        for d in range(hgroups_number):
            H_d = PD[d]  # we get the persistence intervals
            hgroups.append(np.nan_to_num(H_d, posinf=max_eps))  # then we change any inf value with the max_eps

        return hgroups

    @staticmethod
    def homogenyze_diagram(PD):
        '''
        This method obtains a persistence diagram in Gudhi format by transforming a PD from Giotto or Ripser

        :param PD: persistence diagram in one of the previous formats
        :return:
        '''
        if not PersistenceDiagramHelper.is_in_gudhi_format(PD):
            return PersistenceDiagramHelper.ripser_2_useful_pd(PD)

        return PersistenceDiagramHelper.gudhi_2_useful_pd(PD)


    @staticmethod
    def equalize_dimensions(A, B):
        si1 = len(A)
        si2 = len(B)

        max_dim = max(si1, si2)

        A.extend([[[0, 0]]] * (max_dim - si1))
        B.extend([[[0, 0]]] * (max_dim - si2))

        return A, B

    @staticmethod
    def equalize_homology_groups(A, B):
        siA = len(A)
        siB = len(B)
        A = [list(A[i]) for i in range(siA)]
        B = [list(B[i]) for i in range(siB)]

        max_dim = max(siA, siB)
        min_dim = min(siA, siB)

        for d in range(min_dim):
            siAd = len(A[d])
            siBd = len(B[d])

            if siBd > siAd:
                A[d].extend([0] * (siBd - siAd))
            elif siAd > siBd:
                B[d].extend([0] * (siAd - siBd))

        _AA = A if min_dim == siA else B
        _BB = B if max_dim == siB else A
        for d in range(min_dim, max_dim):
            _AA.append([0] * len(_BB[d]))

        return _AA, _BB

    @staticmethod
    def equalize_homology_groups_h1(A, B):
        siA = len(A)
        siB = len(B)

        max_dim = max(siA, siB)
        A.extend([0] * (max_dim - siA))
        B.extend([0] * (max_dim - siB))

        return A, B

    @staticmethod
    def equalize_homology_groups_h1_new(A, B):
        # A = list(A)
        # B = list(B)
        siA = len(A)
        siB = len(B)

        abdif = siA - siB
        badif = siB - siA
        max_val = siA if abdif > 0 else siB

        for d in range(siB, abdif):
            B.append(np.zeros(len(A[d])))

        for d in range(siA, badif):
            A.append(np.zeros(len(B[d])))

        for d in range(max_val):
            bdsize = len(B[d]) - len(A[d])
            if bdsize > 0:
                A[d] = np.append(A[d], [np.zeros(bdsize)])
            else:
                B[d] = np.append(B[d], [np.zeros(-bdsize)])

        return A, B

    @staticmethod
    def get_maximum_sizes(iid_PD):
        max_sizes = {}
        for diag in iid_PD:
            for d, hg in enumerate(diag):
                hgcard = len(hg)
                if d not in max_sizes: # missing dimension
                    max_sizes.update({d: hgcard})
                if max_sizes[d] < hgcard:
                    max_sizes[d] = hgcard
        return max_sizes

    @staticmethod
    def get_diagonal_diagram(PD):
        '''
        This method computes the diagonal diag
        to be [[m_i, m_i]  | m_i = (birth_i+death_i)/2, forall (birth_i,death_i) \in PD[d]]

        :param diag1: persistence diagram 1

        :return: a diagonal diag corresponding to diag1
        '''

        mvect = []
        max_dim = len(PD)
        for d in range(max_dim):
            # compute the array [(birth_i+death_i)/2]_{(birth_i, death_i) \in diag1[d]}
            mid = np.sum(PD[d], axis = 1)*0.5

            # create a new array [[midi,midi]]_{midi \in mid}
            mvect.append(np.vstack((mid, mid)).T)

        return mvect

    @staticmethod
    def get_diagonal_diagram_h1(PD):
        '''
        This method computes the diagonal PD
        to be [[m_i, m_i]  | m_i = (birth_i+death_i)/2, forall (birth_i,death_i) \in PD[d]]

        :param PD: persistence diagram 1

        :return: a diagonal diag corresponding to PD
        '''

        # compute the array [(birth_i+death_i)/2]_{(birth_i, death_i) \in PD[1]}
        if len(PD) < 2:
            return np.array([[0,0]])
        mid = np.sum(PD[1], axis=1) * 0.5

        # create a new array [[midi,midi]]_{midi \in mid}
        return np.vstack((mid, mid)).T

