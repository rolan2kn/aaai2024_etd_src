"""
This unit contains the implementation of the Extended Topology (pseudo)Distance (ETD)

ExtendedTopologyDistance: Naive implementation of our ETD that is faster than WD, SWD
FastExtendedTopologyDistance: Implementation of our ETD that leverage on numpy array programming. 
"""""

import numpy as np
from sklearn.metrics import pairwise_distances

from utilities.persistence_diagram_helper import PersistenceDiagramHelper


class ExtendedTopologyDistanceHelper:
    """
    Naive Extended Topology distance helper functions implementation
    """

    @staticmethod
    def padding_zeroes(A, B):
        return PersistenceDiagramHelper.equalize_homology_groups(A, B)

    @staticmethod
    def padding_zeroes_h1(A, B):
        return PersistenceDiagramHelper.equalize_homology_groups_h1(A, B)

    @staticmethod
    def get_B_vector(PD):
        B = []
        hgroups = len(PD)
        for d in range(hgroups):
            blist = []
            for pi in PD[d]:
                blist.append(pi[0])
            B.append(blist)

        return B

    @staticmethod
    def get_M_vector(PD):
        M = []
        hgroups = len(PD)
        for d in range(hgroups):
            mlist = []
            for pi in PD[d]:
                mlist.append((pi[1] + pi[0]) * 0.5)
            M.append(mlist)

        return M

    @staticmethod
    def get_D_vector(PD):
        D = []
        hgroups = len(PD)
        for d in range(hgroups):
            dlist = []
            for pi in PD[d]:
                dlist.append(pi[1])
            D.append(dlist)

        return D

    @staticmethod
    def get_L_vector(PD):
        longevity = []
        hgroups = len(PD)
        for d in range(hgroups):
            dlongevity = []
            for pi in PD[d]:
                dlongevity.append(pi[1] - pi[0])
            longevity.append(sorted(dlongevity, reverse=True))

        return longevity

    @staticmethod
    def get_V_vector(PD, alpha, Pi):
        if alpha is None:
            return ExtendedTopologyDistanceHelper.get_L_vector(PD)
        '''
        It is really hard to beat calling sin and cos explicitly
        https://stackoverflow.com/questions/32397347/is-there-a-fast-way-to-return-sin-and-cos-of-the-same-value-in-python
        '''

        alpha_interval = np.array([np.cos(alpha), np.sin(alpha)])
        longevity = []
        hgroups = len(PD)

        for d in range(hgroups):
            dlongevity = [np.dot(pi, alpha_interval) for pi in PD[d]]
            dlongevity.extend([np.dot(pi, alpha_interval) for pi in Pi[d]])

            longevity.append(sorted(dlongevity, reverse=True))

        return longevity

    @staticmethod
    def get_L_vector_h1(PD):
        dlongevity = []
        if len(PD) < 2:
            return [0]
        for pi in PD[1]:
            dlongevity.append(pi[1] - pi[0])
        return sorted(dlongevity, reverse=True)

    @staticmethod
    def get_V_vector_h1(PD, alpha, Pi):
        if alpha is None:
            return ExtendedTopologyDistanceHelper.get_L_vector_h1(PD)
        '''
        It is really hard to beat calling sin and cos explicitly
        https://stackoverflow.com/questions/32397347/is-there-a-fast-way-to-return-sin-and-cos-of-the-same-value-in-python
        '''

        alpha_interval = np.array([np.cos(alpha), np.sin(alpha)])

        dlongevity = [np.dot(pi, alpha_interval) for pi in PD[1]]
        dlongevity.extend([np.dot(pi, alpha_interval) for pi in Pi])

        return sorted(dlongevity, reverse=True)

    @staticmethod
    def get_dprojection(P):
        proj = []
        for Pj in P:
            Pj1 = []
            for pi in Pj:
                mi = (pi[0] + pi[1]) * 0.5
                Pj1.append([mi, mi])
            proj.append(Pj1)

        return proj

    @staticmethod
    def get_dprojection_h1(P):
        Pj1 = []
        for pi in P[1]:
            mi = (pi[0] + pi[1]) * 0.5
            Pj1.append([mi, mi])

        return np.array(Pj1)

    @staticmethod
    def compute_angle_dict():
        angle_set = {"A2": [np.pi / 4, 3 * np.pi / 4],
                     "A4": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]}

        k = 2
        for i in range(3):
            k *= 2
            # angle_set.update({f"A{k}": [(3 * np.pi / 4) + (j / k * np.pi) for j in range(0, k)]})
            Ak = []
            for j in range(0, k):
                angle = (3 * np.pi / 4) - (j / k * np.pi)
                if angle < 0:
                    angle = np.pi - angle
                Ak.append(angle)
            angle_set.update({f"A{k}": Ak})
        return angle_set

    @staticmethod
    def compute_swd_slices():
        slice_set = {"M1": 1,
                     "M2": 2,
                     "M4": 4,
                     "M8": 8,
                     "M16": 16}
        return slice_set

    @staticmethod
    def get_basic_etd(HA, HB, p):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param HA: Persistence diagram A
        :param HB: Persistence diagram B

        :return: the topological distance
        """
        A = ExtendedTopologyDistanceHelper.get_L_vector(HA)
        B = ExtendedTopologyDistanceHelper.get_L_vector(HB)
        size = len(A)
        AA, BB = ExtendedTopologyDistanceHelper.padding_zeroes(A, B)

        ETD = [np.linalg.norm(np.subtract(np.array(AA[d]), np.array(BB[d])), ord=p) for d in range(size)]

        del A
        del B
        del AA
        del BB

        return ETD

    @staticmethod
    def get_basic_etd_in_H1(HA, HB, p):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param HA: Persistence diagram A
        :param HB: Persistence diagram B

        :return: the topological distance
        """
        A = ExtendedTopologyDistanceHelper.get_L_vector_h1(HA)
        B = ExtendedTopologyDistanceHelper.get_L_vector_h1(HB)
        AA, BB = PersistenceDiagramHelper.equalize_homology_groups_h1(A, B)
        ETD = np.linalg.norm(np.subtract(np.array(AA), np.array(BB)), ord=p)

        del A
        del B
        del AA
        del BB

        return ETD

    @staticmethod
    def get_etd_alpha(P1, P2, p, alpha, P11=None, P22=None):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram 1
        :param P2: Persistence diagram 2

        :return: the topological distance
        """

        ETDA = []
        alpha_size = len(alpha)
        if alpha_size == 0:
            raise Exception("We were expected a non empty angle set")

        if P11 is None:
            P11 = ExtendedTopologyDistanceHelper.get_dprojection(P1)
        if P22 is None:
            P22 = ExtendedTopologyDistanceHelper.get_dprojection(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector(P2, a, P11)
            size = len(AA)
            ETDa = [(np.linalg.norm(np.subtract(np.array(AA[d]), np.array(BB[d])), ord=p)) ** p
                    for d in range(size)]

            ETDA.append(ETDa)
            del AA
            del BB

        '''
        [
         a0=[etd0, etd1, etd2],   --- 
         a1=[etd0', etd1', etd2'] ---  axis = 1
             |      |       |
                 axis = 0
         ]

        axis = 1      
        '''
        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)  # /alpha_size
        return ETD

    @staticmethod
    def get_etd_alpha_in_H1(P1, P2, p, alpha, P11=None, P22=None):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram 1
        :param P2: Persistence diagram 2

        :return: the topological distance
        """

        ETDA = []
        alpha_size = len(alpha)
        if alpha_size == 0:
            raise Exception("We were expected a non empty angle set")

        if P11 is None:
            P11 = ExtendedTopologyDistanceHelper.get_dprojection_h1(P1)
        if P22 is None:
            P22 = ExtendedTopologyDistanceHelper.get_dprojection_h1(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector_h1(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector_h1(P2, a, P11)
            ETDa = (np.linalg.norm(np.subtract(np.array(AA), np.array(BB)), ord=p)) ** p
            ETDA.append(ETDa)
            del AA
            del BB
        '''
        [
         a0=[etd0, etd1, etd2],   --- 
         a1=[etd0', etd1', etd2'] ---  axis = 1
             |      |       |
                 axis = 0
         ]

        axis = 1      
        '''
        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)  # /alpha_size
        return ETD

    @staticmethod
    def get_etd_alpha_metric(P1, P2, metric, p, alpha=None, P11=None, P22=None):
        """
        This method computes topological distance using a defined metric (we support sci distance metrics). The algorithm is the following:

        We support the following metrics:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix
          inputs.
          ['nan_euclidean'] but it does not yet support sparse matrices.

        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
          See the documentation for scipy.spatial.distance for details on these
          metrics. These metrics do not support sparse matrix inputs.

        1. compute Longevity/V vectors AA, BB. It will depends if alpha is None or if an angle set was given
        2. refill logevity/V vectors A, B per dimension accordingly
        3. compute desired distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram A
        :param P2: Persistence diagram B

        :return: the topological distance
        """
        ETDA = []
        if alpha is None:
            alpha = [None]
        if P11 is None:
            P11 = ExtendedTopologyDistanceHelper.get_dprojection(P1)
        if P22 is None:
            P22 = ExtendedTopologyDistanceHelper.get_dprojection(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector(P2, a, P11)

            size = len(AA)
            ETDa = [(pairwise_distances(X=AA[d], Y=BB[d], n_jobs=-1, metric=metric)) ** p for d in range(size)]

            ETDA.append(ETDa)
            del AA
            del BB

        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)
        return ETD

    @staticmethod
    def get_etd_alpha_metric_in_H1(P1, P2, metric, p, P11, P22, alpha=None):
        """
        This method computes topological distance using a defined metric (we support sci distance metrics). The algorithm is the following:

        We support the following metrics:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix
          inputs.
          ['nan_euclidean'] but it does not yet support sparse matrices.

        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
          See the documentation for scipy.spatial.distance for details on these
          metrics. These metrics do not support sparse matrix inputs.

        1. compute Longevity/V vectors AA, BB. It will depends if alpha is None or if an angle set was given
        2. refill logevity/V vectors A, B per dimension accordingly
        3. compute desired distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram A
        :param P2: Persistence diagram B

        :return: the topological distance
        """
        ETDA = []
        if alpha is None:
            alpha = [None]
        if P11 is None:
            P11 = ExtendedTopologyDistanceHelper.get_dprojection_h1(P1)
        if P22 is None:
            P22 = ExtendedTopologyDistanceHelper.get_dprojection_h1(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector_h1(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector_h1(P2, a, P11)

            ETDa = (pairwise_distances(X=AA, Y=BB, n_jobs=-1, metric=metric)) ** p

            ETDA.append(ETDa)
            del AA
            del BB

        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)
        return ETD

class FastExtendedTopologyDistanceHelper:
    """
    Prototype of faster extended topology distance.
    It relies on numpy array programming techniques to gain several orders of magnitude speed ups over naive ETD
    """
    @staticmethod
    def padding_zeroes_new(A, B):
        max_sizes = PersistenceDiagramHelper.get_maximum_sizes([A, B])

        AA = []
        BB = []
        siA = len(A)
        siB = len(B)

        for d in max_sizes:
            padded = np.zeros(max_sizes[d])
            padded2 = np.zeros(max_sizes[d])
            if siA > d:
                padded[:len(A[d])] = A[d]
            if siB > d:
                padded2[:len(B[d])] = B[d]

            AA.append(padded)
            BB.append(padded2)

        return AA, BB

    @staticmethod
    def padding_zeroes_h1_new(A, B):
        max_sizes = PersistenceDiagramHelper.get_maximum_sizes([[A], [B]])

        AA = np.zeros(max_sizes[0])
        BB = np.zeros(max_sizes[0])
        AA[:len(A)] = A
        BB[:len(B)] = B

        return AA, BB

    @staticmethod
    def get_B_vector_new(PD):
        B = []
        hgroups = len(PD)
        for d in range(hgroups):
            blist = (PD[d])[:, 0]
            B.append(blist)

        return B

    @staticmethod
    def get_M_vector_new(PD):
        '''
        This method computes the diagonal diag

        :param PD: persistence diagram

        :return: a diagonal diag corresponding to PD
        '''

        mvect = []
        max_dim = len(PD)
        for d in range(max_dim):
            # compute the array [(birth_i+death_i)/2]_{(birth_i, death_i) \in diag1[d]}
            mid = np.sum(PD[d], axis=1) * 0.5
            mvect.append(mid)

        return mvect

    @staticmethod
    def get_D_vector_new(PD):
        D = []
        hgroups = len(PD)
        for d in range(hgroups):
            # dlist =
            D.append((PD[d])[:, 1])

        return D

    @staticmethod
    def get_L_vector_new(PD):
        longevity = []
        hgroups = len(PD)
        for d in range(hgroups):
            '''
            dlongevity = []
            for pi in PD[d]:
                dlongevity.append(pi[1] - pi[0])
            '''
            dlongevity = np.diff(PD[d], axis=1).T
            # numpy descending sorting
            dlongevity[0][::-1].sort()

            longevity.append(dlongevity[0])

        return longevity


    @staticmethod
    def get_V_vector_new(PD, alpha, Pi):
        if alpha is None:
            return FastExtendedTopologyDistanceHelper.get_L_vector_new(PD)
        '''
        It is really hard to beat calling sin and cos explicitly
        https://stackoverflow.com/questions/32397347/is-there-a-fast-way-to-return-sin-and-cos-of-the-same-value-in-python
        '''

        alpha_interval = np.array([np.cos(alpha), np.sin(alpha)])
        longevity = []
        hgroups = len(PD)

        for d in range(hgroups):
            '''
            dlongevity = np.concatenate(PD[d], Pi[d]) @ alpha_interval is equivalent to:

            dlongevity = [np.dot(pi, alpha_interval) for pi in PD[d]]
            dlongevity.extend([np.dot(pi, alpha_interval) for pi in Pi[d]])

            ----------------------
            longevity.append(dlongevity[0][::-1].sort()) is equivalent to

            longevity.append(sorted(dlongevity, reverse=True))
            '''
            dlongevity = np.concatenate([PD[d], Pi[d]]) @ alpha_interval
            dlongevity[::-1].sort()
            longevity.append(dlongevity)

        return longevity

    @staticmethod
    def get_L_vector_h1_new(PD):
        '''
        dlongevity = []
        for pi in PD[1]:
            dlongevity.append(pi[1] - pi[0])

        Assumptions:
        1. |PD| > 0
        2. |P1| > 0
        3. PD should be a np array and it should not contains np.nan nor np.inf values
        4. PD should be a np array and it should not contains np.nan nor np.inf values
        '''
        dlongevity = np.diff(PD[1], axis=1).T

        # numpy descending sorting
        dlongevity[0][::-1].sort()

        return dlongevity[0]

    @staticmethod
    def get_V_vector_h1_new(PD, alpha, Pi):
        if alpha is None:
            return ExtendedTopologyDistanceHelper.get_L_vector_h1_new(PD)
        '''
        It is really hard to beat calling sin and cos explicitly
        https://stackoverflow.com/questions/32397347/is-there-a-fast-way-to-return-sin-and-cos-of-the-same-value-in-python
        '''
        alpha_interval = np.array([np.cos(alpha), np.sin(alpha)]).T  # transpose the vector

        '''
        dlongevity = np.concatenate(PD[1], Pi[1]) @ alpha_interval is equivalent to:

        dlongevity = [np.dot(pi, alpha_interval) for pi in PD[1]]
        dlongevity.extend([np.dot(pi, alpha_interval) for pi in Pi])

        ----------------------
        longevity.append(dlongevity[0][::-1].sort()) is equivalent to

        longevity.append(sorted(dlongevity, reverse=True))

        Assumptions:
        1. |PD| > 0
        2. |P1| > 0
        3. PD should be a np array and it should not contains np.nan nor np.inf values
        4. PD should be a np array and it should not contains np.nan nor np.inf values 
        '''
        dlongevity = np.concatenate([PD[1], Pi]) @ alpha_interval
        dlongevity[::-1].sort()

        return dlongevity

    @staticmethod
    def get_dprojection_new(P):
        return PersistenceDiagramHelper.get_diagonal_diagram(P)

    @staticmethod
    def get_dprojection_h1_new(P):
        return PersistenceDiagramHelper.get_diagonal_diagram_h1(P)

    @staticmethod
    def compute_angle_dict():
        angle_set = {"A2": [np.pi / 4, 3 * np.pi / 4],
                     "A4": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]}

        k = 2
        for i in range(3):
            k *= 2
            # angle_set.update({f"A{k}": [(3 * np.pi / 4) + (j / k * np.pi) for j in range(0, k)]})
            Ak = []
            for j in range(0, k):
                angle = (3 * np.pi / 4) - (j / k * np.pi)
                if angle < 0:
                    angle = np.pi - angle
                Ak.append(angle)
            angle_set.update({f"A{k}": Ak})
        return angle_set

    @staticmethod
    def compute_swd_slices():
        slice_set = {"M1": 1,
                     "M2": 2,
                     "M4": 4,
                     "M8": 8,
                     "M16": 16}
        return slice_set

    @staticmethod
    def get_basic_etd_new(HA, HB, p):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param HA: Persistence diagram A
        :param HB: Persistence diagram B

        :return: the topological distance
        """
        A = FastExtendedTopologyDistanceHelper.get_L_vector_new(HA)
        B = FastExtendedTopologyDistanceHelper.get_L_vector_new(HB)

        AA, BB = FastExtendedTopologyDistanceHelper.padding_zeroes_new(A, B)

        size = len(AA)
        ETD = [np.linalg.norm(np.subtract(AA[d], BB[d]), ord=p) for d in range(size)]

        del A
        del B
        del AA
        del BB

        return ETD

    @staticmethod
    def get_basic_etd_in_H1_new(HA, HB, p):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param HA: Persistence diagram A
        :param HB: Persistence diagram B

        :return: the topological distance
        """
        A = FastExtendedTopologyDistanceHelper.get_L_vector_h1_new(HA)
        B = FastExtendedTopologyDistanceHelper.get_L_vector_h1_new(HB)
        AA, BB = FastExtendedTopologyDistanceHelper.padding_zeroes_h1_new(A, B)
        ETD = np.linalg.norm(np.subtract(np.array(AA), np.array(BB)), ord=p)

        del A
        del B
        del AA
        del BB

        return ETD

    @staticmethod
    def get_etd_alpha_new(P1, P2, p, alpha, P11=None, P22=None):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram 1
        :param P2: Persistence diagram 2

        :return: the topological distance
        """

        ETDA = []
        alpha_size = len(alpha)
        if alpha_size == 0:
            raise Exception("We were expected a non empty angle set")

        if P11 is None:
            P11 = FastExtendedTopologyDistanceHelper.get_dprojection_new(P1)
        if P22 is None:
            P22 = FastExtendedTopologyDistanceHelper.get_dprojection_new(P2)

        for a in alpha:
            AA = FastExtendedTopologyDistanceHelper.get_V_vector_new(P1, a, P22)
            BB = FastExtendedTopologyDistanceHelper.get_V_vector_new(P2, a, P11)
            size = len(AA)
            ETDa = [(np.linalg.norm(np.subtract(np.array(AA[d]), np.array(BB[d])), ord=p)) ** p
                    for d in range(size)]

            ETDA.append(ETDa)
            del AA
            del BB

        '''
        [
         a0=[etd0, etd1, etd2],   --- 
         a1=[etd0', etd1', etd2'] ---  axis = 1
             |      |       |
                 axis = 0
         ]

        axis = 1      
        '''
        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)  # /alpha_size
        return ETD

    @staticmethod
    def get_etd_alpha_in_H1_new(P1, P2, p, alpha, P11=None, P22=None):
        """
        This method computes topological distance. The algorithm is the following:

        1. compute Longevity vectors A, B
        2. refill logevity vectors A, B per dimension accordingly
        3. compute euclidean distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram 1
        :param P2: Persistence diagram 2

        :return: the topological distance
        """

        ETDA = []
        alpha_size = len(alpha)
        if alpha_size == 0:
            raise Exception("We were expected a non empty angle set")

        if P11 is None:
            P11 = FastExtendedTopologyDistanceHelper.get_dprojection_h1_new(P1)
        if P22 is None:
            P22 = FastExtendedTopologyDistanceHelper.get_dprojection_h1_new(P2)

        for a in alpha:
            AA = FastExtendedTopologyDistanceHelper.get_V_vector_h1_new(P1, a, P22)
            BB = FastExtendedTopologyDistanceHelper.get_V_vector_h1_new(P2, a, P11)
            ETDa = (np.linalg.norm(np.subtract(np.array(AA), np.array(BB)), ord=p)) ** p

            ETDA.append(ETDa)
            del AA
            del BB

        '''
        [
         a0=[etd0, etd1, etd2],   --- 
         a1=[etd0', etd1', etd2'] ---  axis = 1
             |      |       |
                 axis = 0
         ]

        axis = 1      
        '''
        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)  # /alpha_size
        return ETD

    @staticmethod
    def get_etd_alpha_metric_new(P1, P2, metric, p, alpha=None, P11=None, P22=None):
        """
        This method computes topological distance using a defined metric (we support sci distance metrics). The algorithm is the following:

        We support the following metrics:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix
          inputs.
          ['nan_euclidean'] but it does not yet support sparse matrices.

        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
          See the documentation for scipy.spatial.distance for details on these
          metrics. These metrics do not support sparse matrix inputs.

        1. compute Longevity/V vectors AA, BB. It will depends if alpha is None or if an angle set was given
        2. refill logevity/V vectors A, B per dimension accordingly
        3. compute desired distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram A
        :param P2: Persistence diagram B

        :return: the topological distance
        """
        ETDA = []
        if alpha is None:
            alpha = [None]
        if P11 is None:
            P11 = FastExtendedTopologyDistanceHelper.get_dprojection_new(P1)
        if P22 is None:
            P22 = FastExtendedTopologyDistanceHelper.get_dprojection_new(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector_new(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector_new(P2, a, P11)

            size = len(AA)
            ETDa = [(pairwise_distances(X=AA[d], Y=BB[d], n_jobs=-1, metric=metric)) ** p for d in range(size)]

            ETDA.append(ETDa)
            del AA
            del BB

        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)
        return ETD

    @staticmethod
    def get_etd_alpha_metric_in_H1_new(P1, P2, metric, p, P11, P22, alpha=None):
        """
        This method computes topological distance using a defined metric (we support sci distance metrics). The algorithm is the following:

        We support the following metrics:

        - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']. These metrics support sparse matrix
          inputs.
          ['nan_euclidean'] but it does not yet support sparse matrices.

        - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
          'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
          See the documentation for scipy.spatial.distance for details on these
          metrics. These metrics do not support sparse matrix inputs.

        1. compute Longevity/V vectors AA, BB. It will depends if alpha is None or if an angle set was given
        2. refill logevity/V vectors A, B per dimension accordingly
        3. compute desired distance between longevity vectors
        4. return distance

        :param P1: Persistence diagram A
        :param P2: Persistence diagram B

        :return: the topological distance
        """
        ETDA = []
        if alpha is None:
            alpha = [None]
        if P11 is None:
            P11 = FastExtendedTopologyDistanceHelper.get_dprojection_h1_new(P1)
        if P22 is None:
            P22 = FastExtendedTopologyDistanceHelper.get_dprojection_h1_new(P2)

        for a in alpha:
            AA = ExtendedTopologyDistanceHelper.get_V_vector_h1_new(P1, a, P22)
            BB = ExtendedTopologyDistanceHelper.get_V_vector_h1_new(P2, a, P11)

            ETDa = (pairwise_distances(X=AA, Y=BB, n_jobs=-1, metric=metric)) ** p

            ETDA.append(ETDa)
            del AA
            del BB

        ETD = np.linalg.norm(np.array(ETDA), ord=p, axis=0)
        return ETD

if __name__ == "__main__":
    from utilities.directoy_helper import DirectoryHelper
    from utilities.tda_helper import TDAHelper
    import time

    def perform_function_comparison(train_pdiags,
                                    function_old,
                                    function_new,
                                    vector_name="L",
                                    angle=None, h1 = False, precomputed = False):
        time_old = 0
        time_new = 0
        total_new = total_old = 0
        result_eq = 0
        for id, diag in enumerate(train_pdiags):
            try:

                if angle is None:
                    t1 = time.time_ns()
                    var1 = function_old(diag)
                    t2 = time.time_ns() - t1
                    t3 = time.time_ns()
                    var2 = function_new(diag)
                    t4 = time.time_ns() - t3
                else:
                    if not h1:
                        dproj = FastExtendedTopologyDistanceHelper.get_dprojection_new(diag)

                        if precomputed:
                            dprojj = []
                            pdsize = len(diag)
                            for d in range(pdsize):
                                dprojj.append(np.concatenate([diag[d], dproj[d]]))
                    else:
                        if len(diag) < 2:
                            diag.append([[0,0]])
                        dproj = FastExtendedTopologyDistanceHelper.get_dprojection_h1_new(diag)
                        dprojj = []
                        dprojj.append(diag[0])
                        dprojj.append([[0,0]])
                        if precomputed:
                            dprojj[1] = np.concatenate([diag[1], dproj])

                    t1 = time.time_ns()
                    var1 = function_old(diag, angle, dproj)
                    t2 = time.time_ns() - t1
                    if not precomputed:
                        t3 = time.time_ns()
                        var2 = function_new(diag, angle, dproj)
                        t4 = time.time_ns() - t3
                    else:
                        t3 = time.time_ns()
                        var2 = function_new(dprojj, angle)
                        t4 = time.time_ns() - t3
                time_old += t2
                time_new += t4
                if t4 < t2:
                    total_new += 1
                elif t2 < t4:
                    total_old += 1
                # print(f"results {vector_name}1 in time {t2} ns and {vector_name}2 in time {t4} ns\n")
                eq = 0
                for i in range(len(var1)):
                    # print(f"{vector_name}1[{i}] == {vector_name}2[{1}]: {np.array_equal(np.array(var1[i]), var2[i])} ")
                    if np.array_equal(np.round(var1[i], 6), np.round(var2[i], 6)):
                        eq += 1
                result_eq += 1 if eq == len(var1) else 0
            except Exception as e:
                print(f"Error {e} in diag id {id}")
                raise e

        total = len(train_pdiags)
        new_avg = time_new / total if total > 0 else np.inf
        old_avg = time_old / total if total > 0 else np.inf
        print(f"{vector_name}1 and {vector_name}2 were equal: {(result_eq * 100) / total} %  \n")

        print(f"{vector_name}1 vector was faster: {(total_old * 100) / total} % with {old_avg} avg \n")
        print(f"numpy {vector_name}2 vector was faster: {(total_new * 100) / total} % with {new_avg} avg \n")

    def perform_etd_comparison(train_pdiags,
                                    function_old,
                                    function_new,
                                    function_name="ETD",
                                    angle=None, h1 = False):
        time_old = 0
        time_new = 0
        total_new = total_old = 0
        result_eq = 0
        p = 2
        size = len(train_pdiags)
        if size == 0:
            return
        for i in range(size):
            diagi = train_pdiags[i]
            for j in range(i+1,size):
                diagj = train_pdiags[j]

                if angle is None:
                    t1 = time.time_ns()
                    var1 = function_old(diagi, diagj, p)
                    t2 = time.time_ns() - t1
                    t3 = time.time_ns()
                    var2 = function_new(diagi, diagj, p)
                    t4 = time.time_ns() - t3
                else:
                    t1 = time.time_ns()
                    var1 = function_old(diagi, diagj, p, angle)
                    t2 = time.time_ns() - t1
                    t3 = time.time_ns()
                    var2 = function_new(list(diagi), list(diagj), p, angle)
                    t4 = time.time_ns() - t3
                time_old += t2
                time_new += t4
                if t4 < t2:
                    total_new += 1
                elif t2 < t4:
                    total_old += 1
                eq = 0
                if not h1:
                    for i in range(len(var1)):
                        if np.round(var1[i], 6) == np.round(var2[i], 6):
                            eq += 1
                    result_eq += 1 if eq == len(var1) else 0
                else:
                    if np.round(var1, 6) == np.round(var2, 6):
                        result_eq += 1

        total = size * (size-1) * 0.5
        new_avg = time_new / total if total > 0 else np.inf
        old_avg = time_old / total if total > 0 else np.inf
        print(f"{function_name}1 and {function_name}2 were equal: {(result_eq * 100) / total} %  \n")

        print(f"{function_name}1 distance was faster: {(total_old * 100) / total} % with {old_avg} avg \n")
        print(f"numpy {function_name}2 distance was faster: {(total_new * 100) / total} % with {new_avg} avg \n")

    def perform_padding_comparison(train_pdiags,
                                   pdz,
                                   pdznew,
                                   function_name="pad", h1=False):
        time_old = 0
        time_new = 0
        total_new = total_old = 0
        result_eq = 0
        p = 2
        counter = 0
        train_size = len(train_pdiags)
        for i in range(train_size):
            diagi = train_pdiags[i]
            Li = FastExtendedTopologyDistanceHelper.get_L_vector_new(diagi)
            Lii = [list(Li[d]) for d in range(len(Li))]
            if len(Lii) < 2:
                Lii.append([0])
                Li.append(np.zeros(1))
            for j in range(i+1,train_size):
                counter += 1
                diagj = train_pdiags[j]
                Lj = FastExtendedTopologyDistanceHelper.get_L_vector_new(diagj)
                Ljj = [list(Lj[d]) for d in range(len(Lj))]
                if len(Ljj) < 2:
                    Ljj.append([0])
                    Lj.append(np.zeros(1))
                if not h1:
                    t1 = time.time_ns()
                    AA,BB = pdz(Lii, Ljj)
                    t2 = time.time_ns() - t1
                    t3 = time.time_ns()
                    AA1,BB1 = pdznew(Li, Lj)
                    t4 = time.time_ns() - t3
                else:
                    t1 = time.time_ns()
                    AA, BB = pdz(Lii[1], Ljj[1])
                    t2 = time.time_ns() - t1
                    t3 = time.time_ns()
                    AA1, BB1 = pdznew(Li[1], Lj[1])
                    t4 = time.time_ns() - t3

                time_old += t2
                time_new += t4
                if t4 < t2:
                    total_new += 1
                elif t2 < t4:
                    total_old += 1
                eq = 0
                if not h1:
                    size = len(AA)
                    for d in range(size):

                        if np.array_equal(np.round(AA[d], 6), np.round(AA1[d], 6)):
                            eq += 1
                        if np.array_equal(np.round(BB[d], 6), np.round(BB1[d], 6)):
                            eq += 1
                    result_eq += 1 if eq == size * 2 else 0
                else:
                    if np.array_equal(np.round(AA, 6), np.round(AA1, 6)):
                        eq += 1
                    if np.array_equal(np.round(BB, 6), np.round(BB1, 6)):
                        eq += 1

                    result_eq += 1 if eq == 2 else 0

        total = train_size * (train_size-1) * 0.5
        print(f"couter {counter} total {total}")
        new_avg = time_new / total if total > 0 else np.inf
        old_avg = time_old / total if total > 0 else np.inf
        # print(f"{function_name} and {function_name}_new were equal: {(result_eq * 100) / total} %  \n")

        print(f"{function_name} was faster: {(total_old * 100) / total} % with {old_avg} avg \n")
        print(f"numpy {function_name}_new was faster: {(total_new * 100) / total} % with {new_avg} avg \n")

    def perform_L_h1_comparison(train_pdiags):
        '''
        train_pdiags,
                                    function_old,
                                    function_new,
                                    vector_name="L",
                                    angle=None, h1 = False

        :param train_pdiags:
        :return:
        '''
        print("~~**~~**~~**~~**~~** get_L_vector_h1 **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=train_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_L_vector_h1,
                                    function_new=FastExtendedTopologyDistanceHelper.get_L_vector_h1_new,
                                    vector_name="L")

    def perform_L_comparison(train_pdiags):
        '''
        train_pdiags,
                                    function_old,
                                    function_new,
                                    vector_name="L",
                                    angle=None, h1 = False

        :param train_pdiags:
        :return:
        '''
        print("~~**~~**~~**~~**~~** get_L_vector **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=train_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_L_vector,
                                    function_new=FastExtendedTopologyDistanceHelper.get_L_vector_new,
                                    vector_name="L")

    def perform_M_comparison(all_pdiags):
        print("~~**~~**~~**~~**~~** get_M_vector **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_M_vector,
                                    function_new=FastExtendedTopologyDistanceHelper.get_M_vector_new,
                                    vector_name="M")

    def perform_B_comparison(all_pdiags):
        print("~~**~~**~~**~~**~~** get_B_vector **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_B_vector,
                                    function_new=FastExtendedTopologyDistanceHelper.get_B_vector_new,
                                    vector_name="B")


    def perform_D_comparison(all_pdiags):
        print("~~**~~**~~**~~**~~** get_D_vector **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_D_vector,
                                    function_new=FastExtendedTopologyDistanceHelper.get_D_vector_new,
                                    vector_name="D")

    def perform_proj_comparison(all_pdiags):
        print("~~**~~**~~**~~**~~** get_dprojection **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_dprojection,
                                    function_new=FastExtendedTopologyDistanceHelper.get_dprojection_new,
                                    vector_name="Proj")

    def perform_proj_h1_comparison(all_pdiags):
        print("~~**~~**~~**~~**~~** get_dprojection_h1 **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_dprojection_h1,
                                    function_new=FastExtendedTopologyDistanceHelper.get_dprojection_h1_new,
                                    vector_name="H1Proj")

    def perform_V_comparison(all_pdiags, angle):
        print(f"~~**~~**~~**~~**~~** get_v_vector angle {angle}**~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_V_vector,
                                    function_new=FastExtendedTopologyDistanceHelper.get_V_vector_new,
                                    vector_name="V",
                                    angle=angle)

    def perform_V_comparison_h1(all_pdiags, angle):
        print(f"~~**~~**~~**~~**~~** get_v_vector_h1 angle {angle} **~~**~~**~~**~~**")
        perform_function_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_V_vector_h1,
                                    function_new=FastExtendedTopologyDistanceHelper.get_V_vector_h1_new,
                                    vector_name="V",
                                    angle=angle, h1=True)

    def perform_padding_zeroes_comparison(all_pdiags):
        print(f"~~**~~**~~**~~**~~** padding_zeroes **~~**~~**~~**~~**")
        perform_padding_comparison(train_pdiags=all_pdiags,
                                   pdz=ExtendedTopologyDistanceHelper.padding_zeroes,
                                   pdznew=FastExtendedTopologyDistanceHelper.padding_zeroes_new,
                                   function_name="PaddingZeroes")

    def perform_padding_zeroes_h1_comparison(all_pdiags):
        print(f"~~**~~**~~**~~**~~** padding_zeroes **~~**~~**~~**~~**")
        perform_padding_comparison(train_pdiags=all_pdiags,
                                   pdz=ExtendedTopologyDistanceHelper.padding_zeroes_h1,
                                   pdznew=FastExtendedTopologyDistanceHelper.padding_zeroes_h1_new,
                                   function_name="PaddingZeroes", h1=True)

    def perform_BETD_comparison(all_pdiags):
        print(f"~~**~~**~~**~~**~~** get_basic_etd **~~**~~**~~**~~**")
        perform_etd_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_basic_etd,
                                    function_new=FastExtendedTopologyDistanceHelper.get_basic_etd_new,
                                    function_name="BasicETD")

    def perform_BETD_h1_comparison(all_pdiags):
        print(f"~~**~~**~~**~~**~~** get_basic_etd_in_h1 **~~**~~**~~**~~**")
        perform_etd_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_basic_etd_in_H1,
                                    function_new=FastExtendedTopologyDistanceHelper.get_basic_etd_in_H1_new,
                                    function_name="BasicETD", h1=True)

    def perform_ETDA_comparison(all_pdiags, angle_set):
        print(f"~~**~~**~~**~~**~~** get_etd_alpha angle {angle_set}**~~**~~**~~**~~**")
        perform_etd_comparison(train_pdiags=all_pdiags,
                                    function_old=ExtendedTopologyDistanceHelper.get_etd_alpha,
                                    function_new=FastExtendedTopologyDistanceHelper.get_etd_alpha_new,
                                    function_name=f"ETDA_{angle_set}", angle=angle_set)



    def perform_ETDA_h1_comparison(all_pdiags, angle_set):
        print(f"~~**~~**~~**~~**~~** get_etd_alpha_H1 angle {angle_set}**~~**~~**~~**~~**")
        perform_etd_comparison(train_pdiags=all_pdiags,
                               function_old=ExtendedTopologyDistanceHelper.get_etd_alpha_in_H1,
                               function_new=FastExtendedTopologyDistanceHelper.get_etd_alpha_in_H1_new,
                               function_name=f"ETDA_{angle_set}", angle=angle_set, h1=True)


    overall_path = "results/SupervisedLearningApp/Shrec07Processor/topological_info"
    # overall_path = "../../aaai2024_etd_src/results/SupervisedLearningApp/OutexProcessor/topological_info"
    # overall_path = "../../aaai2024_etd_src/results/SupervisedLearningApp/Cifar10Processor/topological_info"
    # overall_path = "../../aaai2024_etd_src/results/SupervisedLearningApp/FashionMNistProcessor/topological_info"
    # overall_path = "results/SupervisedLearningApp/Shrec07Processor/topological_info"

    # shrec.compute_distance_matrices()
    pdiag_files_train = DirectoryHelper.get_all_filenames(overall_path,
                                                          file_pattern="pd_train",
                                                          ignore_pattern=".png")
    pdiag_files_train = DirectoryHelper.sort_filenames_by_suffix(pdiag_files_train, sep="_")

    train_pdiags = TDAHelper.get_all_pdiagrams(pdiag_files_train)
    train_pdiags = train_pdiags[:20]

    perform_L_comparison(train_pdiags)
    perform_L_h1_comparison(train_pdiags)
    perform_B_comparison(train_pdiags)
    perform_M_comparison(train_pdiags)
    perform_D_comparison(train_pdiags)
    perform_proj_comparison(train_pdiags)
    perform_proj_h1_comparison(train_pdiags)

    angle_set = ExtendedTopologyDistanceHelper.compute_angle_dict()
    for angle in angle_set["A4"]:
        try:
            perform_V_comparison(train_pdiags, angle=angle)
            perform_V_comparison_h1(train_pdiags, angle=angle)
        except Exception as e:
            print(f"Error {e} with {angle}")

    perform_padding_zeroes_comparison(train_pdiags)
    perform_padding_zeroes_h1_comparison(train_pdiags)

    perform_BETD_comparison(train_pdiags)
    perform_BETD_h1_comparison(train_pdiags)
    perform_ETDA_comparison(train_pdiags, angle_set["A16"])
    perform_ETDA_h1_comparison(train_pdiags, angle_set["A16"])
