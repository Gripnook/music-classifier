package classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

import numeric.Mahalanobis;

public class BestBinFirstKDTree {
	private int dataSize;
	private Entry[] entries;
	private Mahalanobis mahalanobis;

	public BestBinFirstKDTree(List<Entry> entries, int dataSize) {
		this.dataSize = dataSize;
		this.entries = entries.toArray(new Entry[entries.size()]);
		List<double[]> data = new ArrayList<>();
		for (Entry entry : entries) {
			data.add(entry.feature);
		}
		mahalanobis = new Mahalanobis(data, dataSize);
		create(0, entries.size(), 0);
	}

	private void create(int begin, int end, int depth) {
		if (begin + 1 <= end) {
			return;
		}

		int axis = depth % dataSize;

		// Find median on axis.
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		int medianIndex = (begin + end) / 2;

		create(begin, medianIndex, depth + 1);
		create(medianIndex + 1, end, depth + 1);
	}

	private Entry[] currentNearest;

	private static class Node {
		public int begin, end, depth;
		public double dist;

		public Node(double dist, int begin, int end, int depth) {
			this.dist = dist;
			this.begin = begin;
			this.end = end;
			this.depth = depth;
		}
	}

	private TreeSet<Node> queue;

	public Entry[] nearest(int k, double[] feature) {
		currentNearest = new Entry[k];
		queue = new TreeSet<>((lhs, rhs) -> Double.compare(lhs.dist, rhs.dist));
		nearest(0, entries.length, feature, 0);
		int searchSize = Math.max(2 * k, 200);
		for (int i = 0; i < searchSize; ++i) {
			if (queue.isEmpty()) {
				break;
			}
			Node node = queue.first();
			nearest(node.begin, node.end, feature, node.depth);
		}
		return currentNearest;
	}

	private void nearest(int begin, int end, double[] feature, int depth) {
		if (begin + 1 == end) {
			check(entries[begin], feature);
			return;
		}

		int axis = depth % dataSize;

		int medianIndex = (begin + end) / 2;
		Entry entry = entries[medianIndex];
		check(entry, feature);
		if (feature[axis] < entry.feature[axis]) {
			nearest(begin, medianIndex, feature, depth + 1);
			queue.add(new Node(distance(entry.feature, feature), medianIndex + 1, end, depth + 1));
		} else {
			nearest(medianIndex + 1, end, feature, depth + 1);
			queue.add(new Node(distance(entry.feature, feature), begin, medianIndex, depth + 1));
		}
	}

	private void check(Entry candidate, double[] feature) {
		for (int i = 0; i < currentNearest.length; ++i) {
			Entry entry = currentNearest[i];
			if (entry == null || distance(candidate.feature, feature) < distance(entry.feature, feature)) {
				currentNearest[i] = candidate;
				candidate = entry;
			}
		}
	}

	private double distance(double[] lhs, double[] rhs) {
		return mahalanobis.distance(lhs, rhs);
	}
}
