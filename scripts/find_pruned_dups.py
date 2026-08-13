import json
import pathlib as pl
import os
import time
import numpy as np

from superneuromat.neuromorphicmodel import SNN
from superneuromat.port.caspian import CaspianImporter

def pruned_subset(path):
	with open(path, "r") as f:
		content = json.load(f)

	nodes_id = sorted([node["id"] for node in content["Nodes"]])
	node_ind_map = {nid: i for i, nid in enumerate(nodes_id)}
	ind_node_map = {v: k for k, v in node_ind_map.items()}

	inputs, outputs = content["Inputs"], content["Outputs"]
	forward_visited = [False] * len(nodes_id)
	backward_visited = [False] * len(nodes_id)

	# connection matrix: matrix[from][to]
	matrix = np.zeros((len(nodes_id), len(nodes_id)))
	for edge in content["Edges"]:
		frm, to = node_ind_map[edge["from"]], node_ind_map[edge["to"]]
		matrix[frm][to] = 1

	def dfs_traverse(matrix, start_ind, visited, forward=True):
		if forward:
			masks = matrix[start_ind, :] > 0
		else:
			masks = matrix[:, start_ind] > 0

		if len(masks) == 0:
			return

		visited[start_ind] = True
		for end_ind, conn in enumerate(masks):
			if not conn or visited[end_ind]:
				continue

			dfs_traverse(matrix, end_ind, visited, forward=forward)

	for _in in inputs:
		dfs_traverse(matrix, node_ind_map[_in], forward_visited, forward=True)

	for out in outputs:
		dfs_traverse(matrix, node_ind_map[out], backward_visited, forward=False)

	pruned = []
	for i, (fv, bv) in enumerate(zip(forward_visited, backward_visited)):
		if fv and bv:
			pruned.append(ind_node_map[i])

	return pruned


def export_pruned(path, export_dir):
	with open(path, "r") as f:
		content = json.load(f)

	subset = pruned_subset(path)

	marked_for_deletion = []
	# Mark certain edges for deletion
	for i, edge in enumerate(content["Edges"]):
		if edge["from"] not in subset or edge["to"] not in subset:
			marked_for_deletion.append(i)

	# Delete marked edges
	for i in reversed(marked_for_deletion):
		del content["Edges"][i]

	marked_for_deletion.clear()
	# Mark node definitions for deletion
	for i, node in enumerate(content["Nodes"]):
		if node["id"] not in subset:
			marked_for_deletion.append(i)

	# Delete marked node definitions
	for i in reversed(marked_for_deletion):
		del content["Nodes"][i]

	marked_for_deletion.clear()
	# Mark input nodes for deletion
	for i, i_id in enumerate(content["Inputs"]):
		if i_id not in subset:
			marked_for_deletion.append(i)

	# Delete marked input nodes
	for i in reversed(marked_for_deletion):
		del content["Inputs"][i]


	marked_for_deletion.clear()
	# Mark output nodes for deletion
	for i, o_id in enumerate(content["Outputs"]):
		if o_id not in subset:
			marked_for_deletion.append(i)

	# Delete marked output nodes
	for i in reversed(marked_for_deletion):
		del content["Outputs"][i]

	export_path = f"{export_dir}/{pl.Path(path).name}"
	with open(export_path, "w") as f:
		json.dump(content, f, indent=2)


def count_same_ones(net_dict):
	same_nodes = {}
	for k, v in net_dict.items():
		sn_key = f"{v}"
		if sn_key not in same_nodes:
			same_nodes[sn_key] = []

		same_nodes[sn_key].append(k)

	with open("test.json", "w") as f:
		json.dump(same_nodes, f, indent=4)

	with open("sorted-test.json", "w") as f:
		json.dump(dict(sorted({
			k: len(v) for k, v in same_nodes.items()	
		}.items(), key=lambda x: x[1], reverse=True)), f, indent=4)


def compare_nets(pruned_dir):
	net_dict = {}
	match_found = {}
	for filename in os.listdir(pruned_dir):
		path = f"{pruned_dir}/{filename}"
		print(f"Importing {filename}")
		cimp = CaspianImporter(path)
		cimp.snn.allow_signed_leak = True

		with open(path) as f:
			content = json.load(f)

		if len(content["Nodes"]) == 0:
			# Empty, after pruning
			net_dict[filename] = None
		else:
			net_dict[filename] = cimp.network_from_json(content)

		match_found[filename] = False

	potential_matches = {}
	for i, (ka, va) in enumerate(net_dict.items()):
		if match_found[ka]:
			print(f"\t? Skipping {ka}...")
			continue
		
		print(f"{i:5}. Finding a match for '{ka}'")
		match_found[ka] = True
		matches = []
		for kb, vb in net_dict.items():
			if match_found[kb]:
				continue
			
			if ka == kb:
				continue

			if va == vb:
				matches.append(kb)
				match_found[kb] = True
				print(f"\t> {kb}")

		potential_matches[ka] = matches

	with open("potential_matches.json", "w") as f:
		json.dump(potential_matches, f, indent=2)


def prune_all_networks(input_dir, export_dir):
	start = time.time()
	print(f"Going through the networks in '{input_dir}'...")
	for i, file in enumerate(os.listdir(input_dir)):
		print(f"{i:5}. Parsing {file}...")
		export_pruned(f"{input_dir}/{file}", export_dir)

	print(f"Done parsing in {time.time() - start:.3f} s")


if __name__ == "__main__":
	# input_dir = "./260803-112734-farp2-es25x_n6_trial10/260731-133257-farp2-es25_n6_trial10/networks"
	# pruned_dir = "pruned"
	# # prune_all_networks(input_dir, pruned_dir)
	# start = time.time()
	# compare_nets(pruned_dir)
	# print(f"Took {time.time() - start} seconds.")

	with open("potential_matches.json") as f:
		content = json.load(f)

	print(f"# of keys = {len(content)}")

	val_count = {}
	for k, v in content.items():
		val_count[k] = len(v)

	s = sum([c for c in val_count.values()])
	print(f"Repeateds = {s}")