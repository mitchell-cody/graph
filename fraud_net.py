# author: Cody Mitchell
# start_data: 14 Dec 2018
# end_date: in_progress
# Project: fraud detection over networks

import pandas as pd
from uszipcode import SearchEngine
from random import randint
import networkx as nx
import matplotlib.pyplot as plt


class GraphAnalysis():

    def __init__(self):
        self.data = None
        self.graph_data = {}
        self.graph_metrics = {}

    def data_loader(self):
        self.data = pd.read_csv('meta_data.csv', quotechar="'")

    def data_cleaner(self):
        self.data['trans_id'] = self.data.index.values
        self.data['trans_id'] = 'tran' + self.data.trans_id.astype(str)

    def impute_zips(self):
        zip_list = ['79936', '90011', '60629', '90805', '77084', '10467', '93307',
                    '10002', '20001', '22193', '11214', '85364', '75211', '30349']

        search = SearchEngine(simple_zipcode=True)  # set simple_zipcode=False to use rich info database
        ls_zip = []
        dt_zip = {}
        for zip_l in zip_list:
            ls_zip.append(search.by_zipcode(zip_l).to_dict()['zipcode'])
            dt_zip[zip_l] = search.by_zipcode(zip_l).to_dict()
        #todo store this dict of data somewhere to easily access

        self.data['zipcodeOri'] = self.data.zipcodeOri.apply(lambda x: ls_zip[randint(0, len(ls_zip)-1)])
        self.data['zipMerchant'] = self.data.zipMerchant.apply(lambda x: ls_zip[randint(0, len(ls_zip)-1)])

    def graph_nodes(self):
        zip_nodes = pd.concat([self.data.zipcodeOri, self.data.zipMerchant])
        fraud_lab = pd.Series('confirmed_fraud')
        #todo ensure that nothing is removed in terms of cross category duplication

        #todo change this to graph
        self.graph_data['nodes'] = pd.concat([zip_nodes, self.data.customer, self.data.merchant, self.data.trans_id,
                                             self.data.category, self.data.age, self.data.gender,fraud_lab]).\
            drop_duplicates()

    def graph_edges(self):
        edge_mt = self.data[['merchant', 'trans_id']].drop_duplicates()
        edge_mt.columns = [0, 1]
        edge_mz = self.data[['merchant', 'zipMerchant']].drop_duplicates()
        edge_mz.columns = [0, 1]
        edge_ct = self.data[['customer', 'trans_id']].drop_duplicates()
        edge_ct.columns = [0, 1]
        edge_cz = self.data[['customer', 'zipcodeOri']].drop_duplicates()
        edge_cz.columns = [0, 1]
        # how to handle more than one entry?
        # how to handle conflicting entries?
        edge_ca = self.data[['customer', 'age']].drop_duplicates()
        edge_ca.columns = [0, 1]
        edge_cg = self.data[['customer', 'gender']].drop_duplicates()
        edge_cg.columns = [0, 1]
        edge_tc = self.data[['trans_id', 'category']].drop_duplicates()
        edge_tc.columns = [0, 1]
        edge_tf = self.data[['trans_id', 'fraud']].drop_duplicates()
        edge_tf = edge_tf[edge_tf.fraud == 1]
        edge_tf['fraud'] = 'confirmed_fraud'
        edge_tf.columns = [0, 1]

        all_edges = pd.concat([edge_mt, edge_mz, edge_ct, edge_cz, edge_ca, edge_cg,
                               edge_tc, edge_tf], ignore_index=True)

        self.graph_data['edges'] = list(zip(all_edges[0], all_edges[1]))

    def graph_creator(self):
        # modeling this as a multigraph will enable multi connections between nodes for duplications
        self.graph_data['graph'] = nx.Graph()
        print('generating', len(self.graph_data['nodes']), 'total nodes')
        self.graph_data['graph'].add_nodes_from(self.graph_data['nodes'])
        print('creating', len(self.graph_data['edges']), 'edges')
        self.graph_data['graph'].add_edges_from(self.graph_data['edges'])

    def graph_plotter(self):
        query_node = 'tran1022'
        list_pos = [i for i, v in enumerate(self.graph_data['edges'])
                    if v[0] == query_node or v[1] == query_node]
        query_nodes = []
        for pos in list_pos:
            query_nodes.append(self.graph_data['edges'][pos][0])
            query_nodes.append(self.graph_data['edges'][pos][1])
        query_nodes = set(query_nodes)
        print('total first degree connections:', len(query_nodes))

        nx.draw(self.graph_data['graph'].subgraph(query_nodes), with_labels=True)
        plt.show()

        #todo these queries are huge due to shared similarities with gender, etc.
        # ideally we can shift this to a cypher query where we can define transaction route
        list_pos = [i for i, v in enumerate(self.graph_data['edges'])
                    for q_node in query_nodes
                    if v[0] == query_node or v[1] == q_node]
        query_nodes = []
        for pos in list_pos:
            query_nodes.append(self.graph_data['edges'][pos][0])
            query_nodes.append(self.graph_data['edges'][pos][1])
        query_nodes = set(query_nodes)
        print('total second degree connections:', len(query_nodes))

    def graph_metrics(self):
        self.graph_metrics['degree_cent'] = nx.algorithms.degree_centrality(self.graph_data['graph'])
        self.graph_metrics['fraud_path'] = nx.algorithms.single_source_shortest_path(self.graph_data['graph'],
                                                                                        'confirmed_fraud')
        self.graph_metrics['jaccard_coef'] = nx.algorithms.jaccard_coefficient(self.graph_data['graph'])
        #ls_n1 = []
        #ls_n2 = []
        #ls_val = []
        #for n1, n2, val in self.graph_metrics['jaccard_coef']:
        #    ls_n1.append(n1)
        #    ls_n2.append(n2)
        #    ls_val.append(val)

        # core_number
        # PageRank

    def send_to_neo4j(pd_df):
        # todo rework this connector to neo
        # todo explore best options for communicating between neo and python
        # https://neo4j.com/developer/python/
        # create node for each Doc and send to dict for call later
        unique = pd_df.File.unique()
        doc_nodes = {}
        for uq in unique:
            uq_trim = re.sub(".pdf", "", uq)
            auditor = pd_df.loc[pd_df['File'] == uq].Auditor
            a = Node(auditor.values[0], name=uq_trim)
            doc_nodes["aud_{0}".format(uq_trim)] = a
        print("audits")
        # create node for each entity and send to dict for call later
        unique = pd_df.Entity.unique()
        ent_nodes = {}
        for uq in unique:
            if len(uq) > 0:
                uq_trim = re.sub(" ", "_", uq)
                lab = pd_df.loc[pd_df['Entity'] == uq].Label
                a = Node(lab.values[0], name=uq_trim)
                ent_nodes["ent_{0}".format(uq_trim)] = a
        print("entities")
        # iterate over all entities, calling the created doc and entity to create relationships
        tx = Graph(password="mmmdata").begin()
        print("connect")
        for index, row in pd_df.iterrows():
            try:
                doc = "aud_" + re.sub(".pdf", "", row.File)
                doc = doc_nodes[doc]
                ent = "ent_" + re.sub(" ", "_", row.Entity)
                ent = ent_nodes[ent]
                ab = Relationship(doc, "Contains", ent)
                tx.create(ab)
            except:
                print(index, row)
        tx.commit()

    def run_r_script(self):
        self.data.to_csv('bnlearn_dat.csv', sep='\t')
        # need to convert integers

        # run bn_struct.R
        bnlearn_arcs = pd.read_csv('bnlearn_arcs.csv', sep='\t')



from libpgm.nodedata import NodeData
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner

# generate some data to use
nd = NodeData()
nd.load(json_dat)    # an input file
skel = GraphSkeleton()
skel.load("../tests/unittestdict.txt")
skel.toporder()
bn = DiscreteBayesianNetwork(skel, nd)
data = bn.randomsample(200)
