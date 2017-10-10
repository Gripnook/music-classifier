package classifier;

import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

public class BestBinFirstKDTree {
	private int dataSize;
	private Entry[] entries;
	private int bins;

	public BestBinFirstKDTree(List<Entry> entries, int dataSize, int bins) {
		this.dataSize = dataSize;
		this.entries = entries.toArray(new Entry[entries.size()]);
		this.bins = bins;
		create(0, this.entries.length, 0);
	}

	private void create(int begin, int end, int depth) {
		if (begin == end) {
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
		for (int i = 0; i < bins; ++i) {
			if (queue.isEmpty())
				break;
			Node node = queue.first();
			nearest(node.begin, node.end, feature, node.depth);
		}
		return currentNearest;
	}

	private void nearest(int begin, int end, double[] feature, int depth) {
		if (begin == end)
			return;

		int axis = depth % dataSize;

		int medianIndex = (begin + end) / 2;
		Entry entry = entries[medianIndex];
		check(entry, feature);
		if (feature[axis] < entry.feature[axis]) {
			nearest(begin, medianIndex, feature, depth + 1);
			if (crosses(entry, feature, axis)) {
				queue.add(new Node(distance(entry.feature, feature), medianIndex + 1, end, depth + 1));
			}
		} else {
			nearest(medianIndex + 1, end, feature, depth + 1);
			if (crosses(entry, feature, axis)) {
				queue.add(new Node(distance(entry.feature, feature), begin, medianIndex, depth + 1));
			}
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

	private boolean crosses(Entry median, double[] feature, int axis) {
		for (Entry entry : currentNearest) {
			if (entry == null || Math.abs(feature[axis] - median.feature[axis]) < distance(feature, entry.feature))
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
