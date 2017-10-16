package classifier.tree;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import classifier.Entry;
import classifier.SongClassifier;
import main.Genre;
import numeric.Plurality;

/**
 * A classifier which bases its decisions on a plurality vote of randomized
 * decision trees. It creates a set of N trees, each trained on a subset of the
 * data, and selects the most likely genre based on votes from each tree.
 *
 * @author Andrei Purcarus
 *
 */
public class DecisionForestClassifier implements SongClassifier {
	private int numTrees;
	private List<Entry> entries = new ArrayList<>();
	private List<DecisionTree> trees = null;
	private Random rng = new Random();

	public DecisionForestClassifier(int numTrees) {
		this.numTrees = numTrees;
	}

	@Override
	public void add(List<double[]> song, Genre genre) {
		for (double[] feature : song) {
			entries.add(new Entry(feature, genre));
		}
	}

	@Override
	public void train() {
		trees = new ArrayList<>();
		int subsetSize = 1 << 16;
		for (int i = 0; i < numTrees; ++i) {
			// Sample a random subset of the data with replacement.
			List<Entry> data = new ArrayList<>();
			for (int j = 0; j < subsetSize; ++j) {
				int index = rng.nextInt(entries.size());
				data.add(entries.get(index));
			}
			trees.add(new DecisionTree(data));
		}
	}

	@Override
	public Genre classify(List<double[]> song) {
		Plurality<Genre> plurality = new Plurality<>();
		for (DecisionTree tree : trees) {
			plurality.add(classify(tree, song));
		}
		return plurality.vote();
	}

	private Genre classify(DecisionTree tree, List<double[]> song) {
		Plurality<Genre> plurality = new Plurality<>();
		for (double[] feature : song) {
			plurality.add(tree.classify(feature));
		}
		return plurality.vote();
	}

	@Override
	public void clear() {
		entries = new ArrayList<>();
		trees = null;
	}
}
