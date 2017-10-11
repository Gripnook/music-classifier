package classifier;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import main.Genre;
import main.Song;
import numeric.Plurality;

public class DecisionTree {
	private static final Genre[] GENRES = Genre.class.getEnumConstants();
	private static final Map<Genre, Integer> GENRE_INDICES = new HashMap<>();
	static {
		for (int i = 0; i < GENRES.length; ++i) {
			GENRE_INDICES.put(GENRES[i], i);
		}
	}

	private static final int FEATURES = (int) Math.sqrt(Song.FEATURES);

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

		int axis = split(begin, end);
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		double minImpurity = Double.MAX_VALUE;
		int splitIndex = (begin + end) / 2;
		int[] lessCounts = new int[GENRES.length];
		int[] greaterCounts = new int[GENRES.length];
		for (int i = begin; i < end; ++i) {
			++greaterCounts[GENRE_INDICES.get(entries[i].genre)];
		}

		for (int i = begin; i < end; ++i) {
			double impurity = impurity(lessCounts, greaterCounts, begin, i, end);
			if (impurity < minImpurity) {
				minImpurity = impurity;
				splitIndex = i;
			}

			int genreIndex = GENRE_INDICES.get(entries[i].genre);
			--greaterCounts[genreIndex];
			++lessCounts[genreIndex];
		}

		Node node = new Node();
		node.axis = axis;
		node.metric = entries[splitIndex].feature[axis];
		node.left = create(begin, splitIndex, depth + 1);
		node.right = create(splitIndex, end, depth + 1);
		return node;
	}

	private boolean isLeaf(int begin, int end, int depth) {
		return begin + 1 == end || uniformGenre(begin, end);
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

	// Returns the axis over which the impurity is minimized when split optimally.
	private int split(int begin, int end) {
		int bestAxis = 0;
		double minImpurity = Double.MAX_VALUE;
		int[] candidates = new int[FEATURES];
		for (int i = 0; i < FEATURES; ++i) {
			candidates[i] = rng.nextInt(Song.FEATURES);
		}
		for (int axis : candidates) {
			double impurity = split(begin, end, axis);
			if (impurity < minImpurity) {
				minImpurity = impurity;
				bestAxis = axis;
			}
		}
		return bestAxis;
	}

	// Returns the impurity of the best split.
	private double split(int begin, int end, int axis) {
		Arrays.sort(entries, begin, end, (lhs, rhs) -> Double.compare(lhs.feature[axis], rhs.feature[axis]));
		double minImpurity = Double.MAX_VALUE;
		int[] lessCounts = new int[GENRES.length];
		int[] greaterCounts = new int[GENRES.length];
		for (int i = begin; i < end; ++i) {
			++greaterCounts[GENRE_INDICES.get(entries[i].genre)];
		}

		for (int i = begin; i < end; ++i) {
			double impurity = impurity(lessCounts, greaterCounts, begin, i, end);
			if (impurity < minImpurity) {
				minImpurity = impurity;
			}

			int genreIndex = GENRE_INDICES.get(entries[i].genre);
			--greaterCounts[genreIndex];
			++lessCounts[genreIndex];
		}
		return minImpurity;
	}

	// Computes the Gini impurity of the split.
	private double impurity(int[] lessCounts, int[] greaterCounts, int begin, int splitIndex, int end) {
		double result = 0;
		for (int i = 0; i < GENRES.length; ++i) {
			int lessCount = lessCounts[i];
			double pLess = (double) (lessCount) / (splitIndex - begin);
			result += (splitIndex - begin) * pLess * (1 - pLess);

			int greaterCount = greaterCounts[i];
			double pGreater = (double) (greaterCount) / (end - splitIndex);
			result += (end - splitIndex) * pGreater * (1 - pGreater);
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
