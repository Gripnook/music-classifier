package classifier.knn;

import java.util.Arrays;
import java.util.List;

import classifier.Entry;

/**
 * An implementation of a KD tree. This is a data structure that partitions data
 * in N-dimensional space by splitting it along median planes. It can therefore
 * allow for faster lookup of the nearest neighbours by eliminating regions of
 * space where they cannot be located.
 * 
 * @author Andrei Purcarus
 *
 */
public class KDTree {
	private int dataSize;
	private Entry[] entries;

	public KDTree(List<Entry> entries, int dataSize) {
		this.dataSize = dataSize;
		this.entries = entries.toArray(new Entry[entries.size()]);
		create(0, entries.size(), 0);
	}

	private void create(int begin, int end, int depth) {
		if (begin + 1 >= end) {
			return;
		}

		// Chooses the current axis based on the depth.
		int axis = depth % dataSize;

		// Finds the median on the axis and partitions the data such that lower
		// points on the axis lie below the median and higher points lie above
		// it. Note that a partition algorithm can also be used here.
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		int medianIndex = (begin + end) / 2;

		create(begin, medianIndex, depth + 1);
		create(medianIndex + 1, end, depth + 1);
	}

	private Entry[] currentNearest;

	public Entry[] nearest(int k, double[] feature) {
		currentNearest = new Entry[k];
		nearest(0, entries.length, feature, 0);
		return currentNearest;
	}

	private void nearest(int begin, int end, double[] feature, int depth) {
		if (begin == end) {
			return;
		} else if (begin + 1 == end) {
			check(entries[begin], feature);
			return;
		}

		// Chooses the current axis based on the depth.
		int axis = depth % dataSize;

		// Checks the median.
		int medianIndex = (begin + end) / 2;
		Entry entry = entries[medianIndex];
		check(entry, feature);
		if (feature[axis] < entry.feature[axis]) {
			nearest(begin, medianIndex, feature, depth + 1);
			// Only checks the other side of the axis if it is possible that
			// a neighbour nearer than those found so far can be located there.
			if (crosses(entry, feature, axis)) {
				nearest(medianIndex + 1, end, feature, depth + 1);
			}
		} else {
			nearest(medianIndex + 1, end, feature, depth + 1);
			// Only checks the other side of the axis if it is possible that
			// a neighbour nearer than those found so far can be located there.
			if (crosses(entry, feature, axis)) {
				nearest(begin, medianIndex, feature, depth + 1);
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
			if (entry == null || distance(feature, median.feature, axis) < distance(feature, entry.feature)) {
				return true;
			}
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

	private double distance(double[] lhs, double[] rhs, int axis) {
		return Math.abs(lhs[axis] - rhs[axis]);
	}
}
