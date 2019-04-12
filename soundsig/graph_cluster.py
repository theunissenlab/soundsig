from __future__ import division, print_function

import copy

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from sklearn.cluster import KMeans
from plots import multi_plot


class EigenvectorModularity(object):

    def __init__(self, g, heuristic_swap=False, null_weight=1.0):
        self.g = g
        self.dendrogram = None  #dendrogram
        self.is_directed = g.__class__.__name__ == 'DiGraph'
        self.B = None
        self.heuristic_swap = heuristic_swap
        self.null_weight = null_weight

    def compute_modularity_matrix(self, g, null_weight=1.0):
        """
            This method computes the modularity matrix from a graph g. Subclasses have to implement this.
        """
        raise NotImplementedError('Do not use EigenvectorModularity class directly, use one of it\'s subclasses!')

    def get_communities(self, min_size=None):
        """
            Main method, gets a list of communities from the graph specified in constructor. Uses the eigenvector
            method to do a greedy maximization of modularity.
        """

        if self.dendrogram is not None:
            communities = list()
            get_communities_from_dendrogram(self.dendrogram, communities, min_size=min_size)
            Qfinal = compute_modularity_from_matrix(self.g, self.B, communities)
            print('Final Modularity: %0.3f' % Qfinal)
            return communities

        #construct dendrogram that contains graph splits, leaves are communities
        dg = nx.DiGraph()
        dg_root_id = 'root'
        dg.add_node(dg_root_id)
        dg.node[dg_root_id]['subgraph_nodes'] = self.g.nodes()

        #compute modularity matrix (implemented by subclass)
        self.B = self.compute_modularity_matrix(self.g, null_weight=self.null_weight)
        assert self.B is not None
        assert self.B.shape == (self.g.number_of_nodes(), self.g.number_of_nodes())

        #find initial split
        sg1,sg2,Q = eigenvector_partition(self.g, self.B, self.is_directed, self.heuristic_swap)

        if len(sg1) == 0 or len(sg2) == 0:
            print('No initial partition found!')
            return [self.g.nodes()]

        #recursively find all the other partitions
        self.find_subcommunities(sg1, self.g, self.B, dg, dg_root_id, 'a')
        self.find_subcommunities(sg2, self.g, self.B, dg, dg_root_id, 'b')
        self.dendrogram = dg

        #build list of communities from dendrogram
        communities = list()
        get_communities_from_dendrogram(self.dendrogram, communities, min_size=min_size)
        Qfinal = compute_modularity_from_matrix(self.g, self.B, communities)
        print('Final Modularity: %0.3f' % Qfinal)

        return communities


    def find_subcommunities(self, subgraph_nodes, g, B, dg, dg_parent_id, handedness):
        """
            Recursive function for finding splits in subgraphs for the eigenvector modularity method.
        """
        Bg = compute_generalized_modularity_matrix(g, B, subgraph_nodes, is_directed=self.is_directed)
        sg = g.subgraph(subgraph_nodes)
        #prune_graph(sg)
        if len(sg) == 0:
            return

        #append to dendrogram
        #dg_sg_id = subgraph_md5(sg)
        dg_sg_id = '%s->%s' % (dg_parent_id, handedness)
        dg.add_node(dg_sg_id)
        dg.node[dg_sg_id]['subgraph_nodes'] = sg.nodes()
        dg.add_edge(dg_sg_id, dg_parent_id)

        sg1,sg2,Q = eigenvector_partition(sg, Bg, self.is_directed, heuristic_swap=self.heuristic_swap)
        if len(sg1) == 0 or len(sg2) == 0 or Q < 0.0:
            return

        #recursively find other partitions
        self.find_subcommunities(sg1, g, B, dg, dg_sg_id, 'a')
        self.find_subcommunities(sg2, g, B, dg, dg_sg_id, 'b')



class UndirectedModularity(EigenvectorModularity):
    """
        Implementation of eigenvector modularity maximization for undirected weighted graphs.
    """

    def __init__(self, g, heuristic_swap=False, null_weight=1.0):
        EigenvectorModularity.__init__(self, g, heuristic_swap=heuristic_swap, null_weight=null_weight)

    def compute_modularity_matrix(self, g, null_weight=1.0):

        if self.B is not None:
            return self.B

        #compute modularity matrix
        N = g.number_of_nodes()


        #compute total weight across all edges
        total_weights = dict()
        for n1 in g.nodes():
            tw = 0.0
            for n2 in g[n1]:
                tw += g[n1][n2]['weight']
            total_weights[n1] = tw

        M = np.array(total_weights.values()).sum()

        #compute modularity
        B = np.zeros([N, N])
        for i,ni in enumerate(g.nodes()):
            for j,nj in enumerate(g.nodes()):
                wi = total_weights[ni]
                wj = total_weights[nj]
                aij = 0.0
                if nj in g[ni]:
                    aij = g[ni][nj]['weight']
                B[i, j] = aij - null_weight*((wi*wj) / M)

        self.B = B
        return self.B



class LinkRankModularity(EigenvectorModularity):
    """
        Implementation of directed graph modularity maximization from:
        Youngdo Kim, Seung-Woo Son, and Hawoong Jeong,
         "Finding communities in directed networks" PHYSICAL REVIEW E 81, 016103 2010
    """

    def __init__(self, g, heuristic_swap=False, null_weight=1.0, alpha=0.85):
        self.alpha = alpha
        EigenvectorModularity.__init__(self, g, heuristic_swap=heuristic_swap, null_weight=null_weight)

    def compute_modularity_matrix(self, g, null_weight=1.0):

        if self.B is not None:
            return self.B

        self.N = g.number_of_nodes()

        #compute page rank of each node
        self.page_rank = nx.pagerank(g, alpha=self.alpha)

        #compute google and link rank matrix
        self.G, self.L = compute_link_rank_matrix(g, self.page_rank, alpha=self.alpha)

        #compute modularity matrix
        self.B, self.Bnull = compute_link_rank_modularity_matrix(g, self.page_rank, self.L, null_weight=null_weight)

        return self.B



def compute_modularity_from_matrix(g, B, communities):
    """
        Computes the modularity (a scalar) from a modularity matrix and a labeling of each node
         and it's community. g is a networkx graph, B is a modularity matrix, and communities is
         a list of lists, which each sub-list containing a list of nodes within that community.
    """

    #compute total weight
    total_weight = 0.0
    for n1,n2 in g.edges():
        total_weight += g[n1][n2]['weight']

    #make a dictionary of nodes to communities
    node2comm = dict()
    for k,comm in enumerate(communities):
        for n in comm:
            node2comm[n] = k

    #compute modularity
    modularity = 0.0
    for i,n1 in enumerate(g.nodes()):
        for j,n2 in enumerate(g.nodes()):
            if node2comm[n1] == node2comm[n2]:
                modularity += B[i, j]

    if not type(g).__name__ == 'DiGraph':
        modularity /= 2*total_weight

    return modularity

def merge_dendrogram_small_children(dg, node_id, min_size):
    """
        Merge child nodes in a dendrogram with their siblings if they're smaller than min_size.
    """
    child_nodes = dg.predecessors(node_id)

    needs_merge = False
    for cnode in child_nodes:
        if len(dg.node[cnode]['subgraph_nodes']) < min_size:
            needs_merge = True
    if needs_merge:
        #break link with children, effectively leaving only the parent node in the dendrogram
        for cnode in child_nodes:
            dg.remove_edge(cnode, node_id)
    else:
        for cnode in child_nodes:
            merge_dendrogram_small_children(dg, cnode, min_size)

def iterate_dendrogram(dg, node_id, ifunc):
    ifunc(dg, node_id)
    for cnode in dg.predecessors(node_id):
        iterate_dendrogram(dg, cnode, ifunc)


def get_communities_from_dendrogram(dg, communities, min_size=None, node_id='root'):
    """ From a dendrogram, get communities. Communities are the leaves of the dendrogram, or if min_size is
        specified, the deepest one can go in the dendrogram without finding a community smaller than min_size.
        This is a recursive function, communities should be passed in as an empty array and will be filled on return.
    """

    if min_size is not None and node_id == 'root':
        #merge subgraphs with their siblings when they are smaller than min_size
        dg_copy = copy.copy(dg)
        merge_dendrogram_small_children(dg_copy, 'root', min_size)
        dg_to_use = dg_copy
    else:
        dg_to_use = dg

    nodes = dg_to_use.node[node_id]['subgraph_nodes']

    node_children = dg_to_use.predecessors(node_id)
    if len(node_children) == 0:
        #we're in a leaf node
        communities.append(nodes)
    else:
        for child in node_children:
            get_communities_from_dendrogram(dg_to_use, communities, min_size=min_size, node_id=child)


def eigenvector_partition(g, B, is_directed=False, heuristic_swap=False):
    """ Partition a graph into two subgraphs based on the eigenvector modularity method """

    #compute eigenvalues/eigenvectors, find max

    Bused = B
    if is_directed:
        #for directed graphs, there's a "trick". we want to find the eigenvalues of a symmetric matrix, so that
        #there are no complex-valued eigenvalues to screw up the method. we do that by adding the transpose of B
        Bused += B.transpose()

    U,V = np.linalg.eig(Bused) #find eigendecomposition
    kmax = U.argmax()
    umax = U[kmax] #largest eigenvalue
    vmax = V[:, kmax].squeeze() #eigenvector associated with largest eigenvalue

    assert np.imag(umax) == 0.0 #the maximum eigenvalue should not be complex (none of them should!)

    #if all eigenvalues are negative or zero, we cannot partition
    if umax <= 0.0:
        return [],[],0.0

    #find optimal partitioning
    s = np.sign(vmax)
    g1 = list()
    g2 = list()
    for k,n in enumerate(g.nodes()):
        if s[k] > 0.0:
            g1.append(n)
        else:
            g2.append(n)

    #compute modularity
    Q = np.dot(s, np.dot(Bused, s)) / (4*g.number_of_edges())

    if heuristic_swap:
        #swap the community of each node to try and further maximize modularity in a fine-tuning process
        print('Heuristic Swap turned on, starting Q=%0.6f' % Q)
        Qbest = Q
        Qdiff = 0.0
        sbest = s
        nodes_swapped = dict()
        start = True
        while start or Qdiff > 0.0:
            start = False
            Qdiff = 0.0

            #go through each node
            for k,n in enumerate(g.nodes()):

                #if the node has been successfully swapped already, ignore it
                if n in nodes_swapped:
                    continue

                #flip the community of node n
                snew = copy.copy(sbest)
                snew[k] = -sbest[k]
                #recompute the modularity
                Qnew = np.dot(snew, np.dot(Bused, snew)) / (4*g.number_of_edges())
                if Qnew > Qbest:
                    print('swapped node %d, Qold=%0.6f, Qnew=%0.6f' % (n, Qbest, Qnew))
                    #if we do better on modularity, keep this swap
                    Qdiff = Qbest - Qnew
                    Qbest = Qnew
                    sbest = snew
                    nodes_swapped[n] = True

        #recompute the subgraphs
        Q = Qbest
        g1 = list()
        g2 = list()
        for k,n in enumerate(g.nodes()):
            if sbest[k] > 0.0:
                g1.append(n)
            else:
                g2.append(n)

    return g1,g2,Q


def compute_generalized_modularity_matrix(g_parent, B, subgraph_nodes, is_directed=False):

    nodes_parent = g_parent.nodes()
    Ng = len(subgraph_nodes)
    Bg = np.zeros([Ng, Ng])

    for i in range(Ng):
        for j in range(Ng):
            ni = subgraph_nodes[i]
            nj = subgraph_nodes[j]
            i_parent = nodes_parent.index(ni)
            j_parent = nodes_parent.index(nj)
            bij = B[i_parent, j_parent]
            if i == j:
                dsum = 0.0
                for k in range(Ng):
                    k_parent = nodes_parent.index(subgraph_nodes[k])
                    dsum += B[i_parent, k_parent]
                    if is_directed:
                        dsum += B[k_parent, i_parent]
                m = 1.0 - (is_directed*0.5)  #multiplier is 1/2 for directed graphs, 1 for undirected
                bij -= m*dsum
            Bg[i, j] = bij
    return Bg


def compute_link_rank_modularity_matrix(g, page_rank, L, null_weight=1.0):

    N = g.number_of_nodes()
    Bnull = np.zeros([N, N], dtype='float')

    for i,n1 in enumerate(g.nodes()):
        for j,n2 in enumerate(g.nodes()):
            Bnull[i, j] = page_rank[n1]*page_rank[n2]

    B = L - null_weight*Bnull
    return B, Bnull


def compute_google_matrix(g, alpha=0.85):

    N = g.number_of_nodes()
    G = np.zeros([N, N], dtype='float')

    #compute total out weight for each node
    w_out = {}
    for n1 in g.nodes():
        w_out[n1] = 0.0
        for n2,eparams in g[n1].iteritems():
            w_out[n1] += eparams['weight']

    all_edges = g.edges()
    #compute google matrix
    for i,n1 in enumerate(g.nodes()):
        for j,n2 in enumerate(g.nodes()):
            if (n1,n2) in all_edges:
                ai = 1 - int(w_out[n1] > 0.0)
                gij = alpha*(g[n1][n2]['weight'] / w_out[n1]) + (1.0/N)*(alpha*ai + 1 - alpha)
                G[i, j] = gij

    return G


def compute_link_rank_matrix(g, page_rank, alpha=0.85):

    N = g.number_of_nodes()
    G = compute_google_matrix(g, alpha=alpha)
    L = np.zeros([N, N], dtype='float')

    for i,n1 in enumerate(g.nodes()):
        for j,n2 in enumerate(g.nodes()):
            L[i, j] = page_rank[n1] * G[i, j]

    return G, L

def prune_graph(g):
    """ Remove unconnected nodes """
    nodes2remove = []
    for n in g.nodes():
        if len(g[n]) == 0:
            nodes2remove.append(n)
    for n in nodes2remove:
        g.remove_node(n)


class SpectralCluster(object):
    """ Spectral cluster graph g (Azran and Ghahramani 2010) based on the affinity matrix P """

    def __init__(self, g, K=None, Mvals=range(1, 500), weight_param='weight'):
        self.g = g
        self.K = K
        self.Mvals = Mvals
        self.W,self.D,self.P = compute_affinity(self.g, weight_param=weight_param)
        self.compute_spectrum()

    def compute_principle_component(self, v):
        denom = np.dot(v, np.dot(self.D, v))
        return np.dot(np.outer(v, v), self.D) / denom

    def compute_spectrum(self):

        #compute spectrum of P
        eigenvalues,V = np.linalg.eig(self.P)

        #sort the spectrum from highest to lowest eigenvalue
        spectrum = list()
        for k,n in enumerate(self.g.nodes()):
            v = np.real(V[:, k])
            principle_component = self.compute_principle_component(v)
            spectrum.append( (n, np.real(eigenvalues[k]), v, principle_component, np.abs(np.real(eigenvalues[k]))) )
        spectrum.sort(key=operator.itemgetter(-1), reverse=True)
        self.spectrum = spectrum

        #compute the max gap \Delta{M}(M) and max gap index K(M) for each M
        ev = np.array([x[1] for x in spectrum])
        max_gap = np.zeros([len(self.Mvals)])
        max_gap_index = np.zeros([len(self.Mvals)])

        for k,M in enumerate(self.Mvals):
            pspec = ev**M
            pspec_diff = np.array([pspec[j] - pspec[j+1] for j in range(len(ev)-1)])
            max_gap_index[k] = pspec_diff.argmax() + 1
            max_gap[k] = pspec_diff.max()

        #determine the maximum value of M that makes sense
        one_gi = max_gap_index == 1
        if one_gi.sum() > 0:
            Mmax_index = np.where(one_gi)[0].min()
        else:
            Mmax_index = 0
        Mmax = self.Mvals[Mmax_index]

        self.M = np.array(range(1, Mmax+1))
        self.Mmax = Mmax
        self.max_gap = max_gap[:Mmax_index+1]
        self.max_gap_index = max_gap_index[:Mmax_index+1]

    def cluster(self, K=None):
        if K is not None:
            self.K = K

        #determine K if needed
        if self.K is None:
            raise NotImplementedError('Automatic determination of K not implemented!')

        #use K-means clustering on the eigenvectors
        N = len(self.spectrum)
        X = np.zeros([N, self.K-1])
        for k in range(self.K-1):
            eigenvector = self.spectrum[k+1][2]
            X[:, k] = eigenvector

        kmean = KMeans(self.K, init='k-means++', n_init=100, max_iter=500, precompute_distances=True)
        kmean.fit(X)

        #determine cluster membership from results of K-means
        clusters = kmean.predict(X)
        node_clusters = dict()
        unique_clusters = np.unique(clusters)
        for uc in unique_clusters:
            node_clusters[uc] = list()
        for k,cluster in enumerate(clusters):
            n = self.spectrum[k][0]
            node_clusters[cluster].append(n)

        self.node_clusters = node_clusters

        return node_clusters

    def plot(self, markersize=12):

        V = np.array([x[2] for x in self.spectrum]).transpose()
        plt.figure()
        plt.imshow(V, interpolation='nearest', aspect='auto')
        plt.xlabel('Eigenvector')
        plt.ylabel('Node')

        plt.figure()
        plt.imshow(V[:, :self.K], interpolation='nearest', aspect='auto')
        plt.xlabel('Eigenvector')
        plt.ylabel('Node')
        plt.xticks([])
        plt.xticks(range(self.K))

        ev = np.abs(np.array([x[1] for x in self.spectrum]))
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(ev, 'go', markersize=markersize)
        plt.ylabel('Eigenvalue')
        plt.title('M=1')
        plt.ylim(0.0, 1.0+0.1)
        plt.subplot(3, 1, 2)
        plt.plot(ev**2, 'go', markersize=markersize)
        plt.ylabel('Eigenvalue')
        plt.ylim(0.0, 1.0+0.1)
        plt.title('M=2')
        plt.subplot(3, 1, 3)
        plt.plot(ev**self.Mmax, 'go', markersize=markersize)
        plt.ylabel('Eigenvalue')
        plt.title('M=%d' % self.Mmax)
        plt.ylim(0.0, 1.0+0.1)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.M, self.max_gap, 'k-')
        plt.xlabel('M')
        plt.ylabel('$\Delta{M}(M)$')
        plt.subplot(2, 1, 2)
        plt.plot(self.M, self.max_gap_index, 'k-')
        plt.xlabel('M')
        plt.ylabel('K(M)')

        dlist = list()
        for k in range(self.K+1):
            n,eigenvalue,eigenvector,principle_component,abs_eigenvalue = self.spectrum[k]
            dlist.append({'eigenvalue':eigenvalue, 'T':principle_component, 'k':k})
        multi_plot(dlist, plot_principle_component, nrows=3, ncols=3)


def plot_principle_component(pdata, ax):
    T = pdata['T']
    v = pdata['eigenvalue']
    k = pdata['k']
    plt.imshow(T, interpolation='nearest', aspect='auto')
    plt.title('$\lambda_%d$=%0.6f' % (k,v))
    plt.xticks([])
    plt.yticks([])


def get_weight_matrix(g, weight_param='weight'):
    N = len(g.nodes())
    W = np.zeros([N, N])
    node2index = dict()
    for k,n in enumerate(g.nodes()):
        node2index[n] = k
    for n1,n2 in g.edges():
        i = node2index[n1]
        j = node2index[n2]
        w = g[n1][n2][weight_param]
        W[i][j] = w
        W[j][i] = w
    return W


def compute_affinity(g, weight_param='weight'):
    """ Computes the affinity matrix for an undirected graph g. """

    #compute the matrix D = diag(d(1), ..., d(N)), where d(i) is the sum of weights for node i
    N = len(g.nodes())
    W = get_weight_matrix(g, weight_param=weight_param)

    node_weights = np.zeros([N])
    for k,n in enumerate(g.nodes()):
        node_weights[k] = W[k, :].sum()
    znodes = (node_weights == 0.0)
    if znodes.sum() > 0:
        print(node_weights)
        raise ValueError('Cannot compute affinity matrix for graph that has disconnected nodes, or whose weight sum is zero.')

    D = np.diag(node_weights)
    Dinv = np.linalg.inv(D)

    P = np.dot(Dinv, W)
    return W,D,P





