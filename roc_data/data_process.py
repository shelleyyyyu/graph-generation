# -*- coding:UTF-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_with_weight(vertex, edges, edge_width):
	g = nx.Graph()
	#g.add_nodes_from(vertex)
	new_edgelist = []
	for eid, e in enumerate(edges):
		new_edgelist.append((e[0], e[1], edge_width[eid]))
	g.add_weighted_edges_from(new_edgelist)
	return g 

def create_graph(vertex, edges):
	g = nx.Graph()
	g.add_nodes_from(vertex)
	new_edgelist = []
	for eid, e in enumerate(edges):
		new_edgelist.append((e[0], e[1]))
	g.add_edges_from(new_edgelist)
	return g 

def process_graph(all_story):
	graphs = []
	for story in all_story:
		edges = {}
		for k_w in story['keyword']:
			sub_exist = []
			for sid, s in enumerate(story['story']):
				if k_w in s:
					sub_exist.append(sid)
			
			if len(sub_exist) > 1:
				for gp_id, gp in enumerate(sub_exist):
					for second_gp_id, second_gp in enumerate(sub_exist[(gp_id+1):]):
						if (gp, second_gp) not in edges: 
							edges[(str(gp), str(second_gp))] = 1
						else:
							edges[(str(gp), str(second_gp))] = edges[(str(gp), str(second_gp))] +1


		vertex = ['0','1','2','3','4']
		#g = create_graph_with_weight(vertex, edges.keys(), list(edges.values()))
		g = create_graph(vertex, edges.keys())
		graphs.append(g)

	return graphs
    

def get_roc_graph(path):
	all_story = []
	with open (path, 'r') as file:
		stories = file.readlines()
		for story in stories:
			story_dict = {}
			story_dict['title'] = story.split("<EOT>")[0].strip().split()
			story_dict['keyword'] = story.split("<EOT>")[1].split("<EOL>")[0].strip().split()
			story_arr = story.split("<EOT>")[1].split("<EOL>")[1].strip().split('</s>')
			story_new_arr = []
			for line in story_arr:
				if line != '':
					story_new_arr.append(line.strip().split())
			story_dict['story'] = story_new_arr
			all_story.append(story_dict)
	return all_story


