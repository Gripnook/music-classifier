package classifier;

import java.util.Collections;
import java.util.List;

public class KDTree {
	private static class Node {
		public Entry entry;
		public Node left, right;
	}

	private int dataSize;
	private Node root;

	public KDTree(List<Entry> entries, int dataSize) {
		this.dataSize = dataSize;
		root = create(entries, 0);
	}

	private Node create(List<Entry> entries, int depth) {
		if (entries.isEmpty()) {
			return null;
		}

		int axis = depth % dataSize;

		// Find median on axis.
		Collections.sort(entries, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		int medianIndex = entries.size() / 2;
		Entry median = entries.get(medianIndex);

		Node node = new Node();
		node.entry = median;
		node.left = create(entries.subList(0, medianIndex), depth + 1);
		node.right = create(entries.subList(medianIndex + 1, entries.size()), depth + 1);
		return node;
	}

	private Entry[] currentNearest;

	public Entry[] nearest(int k, double[] feature) {
		currentNearest = new Entry[k];
		nearest(root, feature, 0);
		return currentNearest;
	}

	private void nearest(Node node, double[] feature, int depth) {
		int axis = depth % dataSize;

		check(node, feature);
		if (feature[axis] < node.entry.feature[axis]) {
			if (node.left != null) {
				nearest(node.left, feature, depth + 1);
			}
			if (node.right == null)
				return;
			if (crosses(node, feature, axis)) {
				nearest(node.right, feature, depth + 1);
			}
		} else {
			if (node.right != null) {
				nearest(node.right, feature, depth + 1);
			}
			if (node.left == null)
				return;
			if (crosses(node, feature, axis)) {
				nearest(node.left, feature, depth + 1);
			}
		}
	}

	private void check(Node node, double[] feature) {
		Entry candidate = node.entry;
		for (int i = 0; i < currentNearest.length; ++i) {
			Entry entry = currentNearest[i];
			if (entry == null || distance(candidate.feature, feature) < distance(entry.feature, feature)) {
				currentNearest[i] = candidate;
				candidate = entry;
			}
		}
	}

	private boolean crosses(Node node, double[] feature, int axis) {
		for (Entry entry : currentNearest) {
			if (entry == null || Math.abs(feature[axis] - node.entry.feature[axis]) < distance(feature, entry.feature))
				return true;
		}
		return false;
	}

	private double distance(double[] lhs, double[] rhs) {
		double result = 0;
		for (int i = 0; i < dataSize; ++i) {
			result += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
		}
		return Math.sqrt(result);
	}
}
