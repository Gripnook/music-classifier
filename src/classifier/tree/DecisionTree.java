package classifier.tree;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import classifier.Entry;
import main.Genre;
import main.Song;
import numeric.Plurality;

/**
 * A decision tree meant to be used in a randomized decision forest setting. It
 * trains by dividing its data into two parts at each node in such a way as to
 * maximize the information gain of the resulting split. It then assigns a genre
 * to each leaf of the tree. To classify a new feature, it uses the splits in
 * the tree to reach a leaf node and assigns it that genre.
 * 
 * @author Andrei Purcarus
 *
 */
public class DecisionTree {
	private static final Genre[] GENRES = Genre.class.getEnumConstants();
	private static final Map<Genre, Integer> GENRE_INDICES = new HashMap<>();
	static {
		for (int i = 0; i < GENRES.length; ++i) {
			GENRE_INDICES.put(GENRES[i], i);
		}
	}

	// Uses a random subset of the features in each tree to minimize the bias
	// towards features with a strong correlation to the data.
	private static final int FEATURES = (int) (Math.sqrt(Song.FEATURES));

	private static class Node {
		// Internal node parameters.
		Node left = null, right = null;
		int axis = -1;
		double metric = 0;

		// Leaf node parameters.
		Genre genre = null;

		public Node() {

		}

		public Node(Genre genre) {
			this.genre = genre;
		}
	}

	private Node root;
	private Entry[] entries;
	private Random rng = new Random();

	public DecisionTree(List<Entry> entries) {
		this.entries = entries.toArray(new Entry[entries.size()]);
		root = create(0, entries.size(), 0);
	}

	private Node create(int begin, int end, int depth) {
		if (isLeaf(begin, end, depth)) {
			return new Node(plurality(begin, end));
		}

		// Splits the data on the best axis to minimize the entropy.
		int axis = split(begin, end);
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		double minEntropy = Double.MAX_VALUE;
		int splitIndex = (begin + end) / 2;
		int[] lessCounts = new int[GENRES.length];
		int[] greaterCounts = new int[GENRES.length];
		for (int i = begin; i < end; ++i) {
			++greaterCounts[GENRE_INDICES.get(entries[i].genre)];
		}

		double parentEntropy = entropy(begin, end, greaterCounts);

		for (int i = begin; i < end; ++i) {
			double entropy = entropy(begin, i, end, lessCounts, greaterCounts);
			if (entropy < minEntropy) {
				minEntropy = entropy;
				splitIndex = i;
			}

			int genreIndex = GENRE_INDICES.get(entries[i].genre);
			--greaterCounts[genreIndex];
			++lessCounts[genreIndex];
		}

		// If there is no information to gain, we create a leaf node.
		double informationGain = parentEntropy - minEntropy;
		if (informationGain <= 0) {
			return new Node(plurality(begin, end));
		}

		Node node = new Node();
		node.axis = axis;
		node.metric = entries[splitIndex].feature[axis];
		node.left = create(begin, splitIndex, depth + 1);
		node.right = create(splitIndex, end, depth + 1);
		return node;
	}

	private boolean isLeaf(int begin, int end, int depth) {
		return begin + Song.FEATURES >= end || uniformGenre(begin, end);
	}

	private boolean uniformGenre(int begin, int end) {
		Entry entry = entries[begin];
		for (int i = begin; i < end; ++i) {
			if (!entry.genre.equals(entries[i].genre)) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Returns the axis with the split that minimizes the entropy.
	 * 
	 * @param begin
	 * @param end
	 * @return
	 */
	private int split(int begin, int end) {
		int bestAxis = 0;
		double minEntropy = Double.MAX_VALUE;
		int[] candidates = new int[FEATURES];
		for (int i = 0; i < FEATURES; ++i) {
			candidates[i] = rng.nextInt(Song.FEATURES);
		}
		for (int axis : candidates) {
			double entropy = split(begin, end, axis);
			if (entropy < minEntropy) {
				minEntropy = entropy;
				bestAxis = axis;
			}
		}
		return bestAxis;
	}

	/**
	 * Returns the entropy of the best split on the given axis.
	 * 
	 * @param begin
	 * @param end
	 * @param axis
	 * @return
	 */
	private double split(int begin, int end, int axis) {
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		double minEntropy = Double.MAX_VALUE;
		int[] lessCounts = new int[GENRES.length];
		int[] greaterCounts = new int[GENRES.length];
		for (int i = begin; i < end; ++i) {
			++greaterCounts[GENRE_INDICES.get(entries[i].genre)];
		}

		for (int i = begin; i < end; ++i) {
			double entropy = entropy(begin, i, end, lessCounts, greaterCounts);
			if (entropy < minEntropy) {
				minEntropy = entropy;
			}

			int genreIndex = GENRE_INDICES.get(entries[i].genre);
			--greaterCounts[genreIndex];
			++lessCounts[genreIndex];
		}
		return minEntropy;
	}

	/**
	 * Computes the entropy of the given split.
	 * 
	 * @param begin
	 * @param splitIndex
	 * @param end
	 * @param lessCounts
	 * @param greaterCounts
	 * @return
	 */
	private double entropy(int begin, int splitIndex, int end, int[] lessCounts, int[] greaterCounts) {
		double totalSize = end - begin;
		double lessSize = splitIndex - begin;
		double greaterSize = end - splitIndex;
		if (lessSize == 0) {
			return entropy(splitIndex, end, greaterCounts);
		} else if (greaterSize == 0) {
			return entropy(begin, splitIndex, lessCounts);
		} else {
			return (lessSize / totalSize) * entropy(begin, splitIndex, lessCounts)
					+ (greaterSize / totalSize) * entropy(splitIndex, end, greaterCounts);
		}
	}

	/**
	 * Computes the entropy of the given range.
	 * 
	 * @param begin
	 * @param end
	 * @param counts
	 * @return
	 */
	private double entropy(int begin, int end, int[] counts) {
		double result = 0;
		for (int i = 0; i < GENRES.length; ++i) {
			int count = counts[i];
			if (count != 0) {
				double p = (double) (count) / (end - begin);
				result -= p * Math.log(p);
			}
		}
		return result;
	}

	public Genre classify(double[] feature) {
		Node node = root;
		while (node.left != null && node.right != null) {
			if (feature[node.axis] < node.metric) {
				node = node.left;
			} else {
				node = node.right;
			}
		}
		return node.genre;
	}

	private Genre plurality(int begin, int end) {
		Plurality<Genre> plurality = new Plurality<>();
		for (int i = begin; i < end; ++i) {
			plurality.add(entries[i].genre);
		}
		return plurality.vote();
	}
}
