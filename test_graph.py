"""
Test script for co-occurrence graph builder
"""
import pickle
import torch

print("="*60)
print("TESTING CO-OCCURRENCE GRAPH")
print("="*60)

# Test 1: Load graph data
print("\n[Test 1] Loading co-occurrence graph...")
with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
    graph_data = pickle.load(f)

print(f"✓ Successfully loaded cooccurrence_graph.pkl")
print(f"  Keys: {list(graph_data.keys())}")
print(f"  Config: {graph_data['config']}")

# Test 2: Verify graph structure
print("\n[Test 2] Verifying graph structure...")
edge_index = graph_data['edge_index']
edge_weight = graph_data['edge_weight']
edge_dict = graph_data['edge_dict']
config = graph_data['config']

print(f"✓ Graph components:")
print(f"  Edge index shape: {edge_index.shape}")
print(f"  Edge weight shape: {edge_weight.shape}")
print(f"  Edge dict size: {len(edge_dict):,}")
print(f"  Num items: {config['num_items']:,}")
print(f"  Window size: {config['window_size']}")
print(f"  Min count: {config['min_count']}")

# Test 3: Edge consistency
print("\n[Test 3] Checking edge consistency...")
assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
assert edge_index.shape[1] == edge_weight.shape[0], "Edge index and weights should match"
print(f"✓ Edge dimensions are consistent")

# Test 4: Verify undirected graph
print("\n[Test 4] Verifying undirected graph property...")
num_edges = edge_index.shape[1]
# Each edge should appear in both directions (plus self-loops)
edge_set = set()
for i in range(num_edges):
    src = edge_index[0, i].item()
    dst = edge_index[1, i].item()
    edge_set.add((src, dst))

# Check a few edges have reverse
sample_edges = list(edge_dict.keys())[:10]
reverse_count = 0
for (i, j) in sample_edges:
    if (j, i) in edge_set:
        reverse_count += 1

print(f"✓ Sampled 10 edges, found {reverse_count} with reverse direction")
assert reverse_count >= 8, "Most edges should have bidirectional representation"

# Test 5: Self-loops check
print("\n[Test 5] Checking self-loops...")
self_loops = (edge_index[0] == edge_index[1]).sum().item()
print(f"✓ Number of self-loops: {self_loops:,}")
# Should have self-loops for all items + padding
expected_self_loops = config['num_items'] + 1  # +1 for padding
print(f"  Expected: {expected_self_loops:,}")

# Test 6: Edge weight statistics
print("\n[Test 6] Analyzing edge weights...")
non_self_loop_mask = edge_index[0] != edge_index[1]
non_self_weights = edge_weight[non_self_loop_mask]

print(f"✓ Edge weight statistics (excluding self-loops):")
print(f"  Min: {non_self_weights.min().item():.2f}")
print(f"  Max: {non_self_weights.max().item():.2f}")
print(f"  Mean: {non_self_weights.mean().item():.2f}")
print(f"  Median: {non_self_weights.median().item():.2f}")

# Should all be >= min_count
assert non_self_weights.min() >= config['min_count'], f"All weights should be >= {config['min_count']}"

# Test 7: Top co-occurring pairs
print("\n[Test 7] Top 10 co-occurring item pairs...")
top_edges = sorted(edge_dict.items(), key=lambda x: x[1], reverse=True)[:10]
for idx, ((item_i, item_j), count) in enumerate(top_edges, 1):
    print(f"  {idx}. Item {item_i} <-> Item {item_j}: {count} co-occurrences")

# Test 8: Graph connectivity
print("\n[Test 8] Analyzing graph connectivity...")
unique_nodes = torch.unique(edge_index).tolist()
print(f"✓ Unique nodes in graph: {len(unique_nodes):,}")
print(f"  Total items: {config['num_items']:,}")
print(f"  Coverage: {100 * len(unique_nodes) / config['num_items']:.1f}%")

# Test 9: Degree distribution
print("\n[Test 9] Computing degree distribution...")
from collections import defaultdict
degrees = defaultdict(int)

for i in range(edge_index.shape[1]):
    src = edge_index[0, i].item()
    if edge_index[0, i] != edge_index[1, i]:  # Exclude self-loops
        degrees[src] += 1

degree_values = list(degrees.values())
print(f"✓ Degree statistics:")
print(f"  Nodes with edges: {len(degree_values):,}")
print(f"  Min degree: {min(degree_values)}")
print(f"  Max degree: {max(degree_values)}")
print(f"  Avg degree: {sum(degree_values) / len(degree_values):.2f}")

# Test 10: Verify PyTorch Geometric compatibility
print("\n[Test 10] Testing PyTorch Geometric compatibility...")
try:
    import torch_geometric
    from torch_geometric.data import Data
    
    # Create a PyG Data object
    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=config['num_items'] + 1
    )
    
    print(f"✓ PyG Data object created successfully")
    print(f"  Num nodes: {data.num_nodes}")
    print(f"  Num edges: {data.num_edges}")
    print(f"  Has self-loops: {data.has_self_loops()}")
    print(f"  Is undirected: {data.is_undirected()}")
    
except ImportError:
    print("⚠ torch_geometric not installed, skipping PyG compatibility test")

print("\n" + "="*60)
print("✅ ALL GRAPH BUILDER TESTS PASSED!")
print("="*60)
