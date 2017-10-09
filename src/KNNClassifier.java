import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNClassifier implements SongClassifier {
	private int k;
	private List<Entry> songs = new ArrayList<>();
	private KDTree tree = null;

	public KNNClassifier(int k) {
		this.k = k;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song)
			songs.add(new Entry(feature, genre));
	}

	@Override
	public void train() {
		tree = new KDTree(songs);
	}

	@Override
	public Genre classify(List<double[]> song) {
		Map<Genre, Integer> counts = new HashMap<>();

		for (double[] feature : song) {
			Genre genre = classify(feature);
			Integer count = counts.get(genre);
			if (count == null) {
				count = new Integer(0);
			}
			++count;
			counts.put(genre, count);
		}

		int max = 0;
		Genre result = null;
		for (Genre genre : counts.keySet()) {
			int count = counts.get(genre);
			if (count > max) {
				max = count;
				result = genre;
			}
		}
		return result;
	}

	@Override
	public void clear() {
		songs = new ArrayList<>();
		tree = null;
	}

	private Genre classify(double[] feature) {
		Entry[] nearest = tree.nearest(k, feature);

		Map<Genre, Integer> counts = new HashMap<>();
		for (Entry entry : nearest) {
			Integer count = counts.get(entry.genre);
			if (count == null) {
				count = new Integer(0);
			}
			++count;
			counts.put(entry.genre, count);
		}

		int max = 0;
		Genre result = null;
		for (Genre genre : counts.keySet()) {
			int count = counts.get(genre);
			if (count > max) {
				max = count;
				result = genre;
			}
		}

		return result;
	}
}

class Entry {
	public double[] song;
	public Genre genre;

	public Entry(double[] song, Genre genre) {
		this.song = song;
		this.genre = genre;
	}
}

class KDTree {
	private class Node {
		public Entry entry;
		public Node left, right;
	}

	private Node root;

	public KDTree(List<Entry> entries) {
		root = create(entries, 0);
	}

	private Node create(List<Entry> entries, int depth) {
		if (entries.isEmpty()) {
			return null;
		}

		int axis = depth % Song.DATA_SIZE;

		// Find median on axis.
		Collections.sort(entries, (lhs, rhs) -> {
			if (lhs.song[axis] == rhs.song[axis])
				return 0;
			if (lhs.song[axis] < rhs.song[axis])
				return -1;
			else
				return 1;
		});
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
		int axis = depth % Song.DATA_SIZE;

		check(node, feature);
		if (feature[axis] < node.entry.song[axis]) {
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
			if (entry == null || distance(candidate.song, feature) < distance(entry.song, feature)) {
				currentNearest[i] = candidate;
				candidate = entry;
			}
		}
	}

	private boolean crosses(Node node, double[] feature, int axis) {
		for (Entry entry : currentNearest) {
			if (entry == null)
				continue;
			if (Math.abs(feature[axis] - node.entry.song[axis]) < distance(feature, entry.song))
				return true;
		}
		return false;
	}

	private double distance(double[] lhs, double[] rhs) {
		double result = 0;
		for (int i = 0; i < Song.DATA_SIZE; ++i) {
			result += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
		}
		return Math.sqrt(result);
	}
}
